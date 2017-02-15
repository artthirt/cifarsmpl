#include "gpu_train.h"

#include <QDebug>

//////////////////////

void test_void(const gpumat::GpuMat& mat)
{
	ct::Matf test;

	gpumat::convert_to_mat(mat, test);

	qDebug("test first %f", test.ptr()[0]);
}

//////////////////////

gpu_train::gpu_train()
{
	m_init = false;
}

void gpu_train::setConvLayers(const std::vector<int> &layers,
							  std::vector<int> weight_sizes,
							  const ct::Size szA0)
{
	if(layers.empty() || weight_sizes.empty())
		throw new std::invalid_argument("empty parameters");

	m_cnvlayers = layers;
	m_cnvweights = weight_sizes;
	m_szA0 = szA0;

	m_conv.resize(3);
	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].setConvLayers(layers, weight_sizes, szA0);
	}

	m_init = false;
}

void gpu_train::setMlpLayers(const std::vector<int> &layers)
{
	if(layers.empty())
		throw new std::invalid_argument("empty parameters");

	m_layers = layers;

	m_init = false;
}

void gpu_train::setAlpha(double alpha)
{
	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].setAlpha(alpha);
	}
	m_optim.setAlpha(alpha);
}

void gpu_train::init()
{
	if(m_init)
		return;

	if(m_layers.empty() || m_cnvlayers.empty() || m_cnvweights.empty())
		throw new std::invalid_argument("empty arguments");

	//// 1

	{
		for(size_t i = 0; i < m_conv.size(); ++i){
			m_conv[i].init();
		}
		qDebug("CNV: ouput matrices = %d", m_conv[0].outputMatrices() * m_conv.size());
	}

	//// 2

	{
		m_mlp.resize(m_layers.size());

		int input = m_conv[0].outputFeatures();
		input *= m_conv.size();

		qDebug("MLP: input features = %d", input);

		for(size_t i = 0; i < m_mlp.size(); ++i){
			gpumat::mlp& mlp = m_mlp[i];
			int output = m_layers[i];

			mlp.init(input, output, gpumat::GPU_FLOAT);

			input = output;
		}
	}

	m_optim.init(m_mlp);

	m_init = true;
}

bool gpu_train::isInit() const
{
	return m_init;
}

uint gpu_train::iteration() const
{
	return m_optim.iteration();
}

double gpu_train::getL2(const ct::Matf &yp, const ct::Matf &y)
{
	if(yp.empty() || y.empty() || yp.rows != y.rows)
		return -1.;

	gpumat::convert_to_gpu(yp, m_gyp);

	gpumat::convert_to_gpu(y, m_y_ind2);

	gpumat::subIndOne(m_gyp, m_y_ind2, m_tsub);

	gpumat::elemwiseSqr(m_tsub, m_tsub);

	//test_void(m_tsub);

	gpumat::reduce(m_tsub, m_red);

	ct::Matf red;

	gpumat::convert_to_mat(m_red, red);

	double l2 = red.at(0, 0);

	return l2 / y.rows;
}

void gpu_train::forward(const std::vector<ct::Matf> &X, ct::Matf &a_out, bool use_drop, double p, bool use_ret)
{
	if(X.empty())
		return;

	m_XsIn.resize(X.size());

	for(size_t i = 0; i < X.size(); ++i){
		gpumat::convert_to_gpu(X[i], m_XsIn[i]);
	}

	gpumat::GpuMat *retA = nullptr;

	forward(m_XsIn, &retA, use_drop, p);

	if(use_ret && retA){
		gpumat::convert_to_mat(*retA, a_out);
	}
}

void gpu_train::forward(const std::vector<gpumat::GpuMat> &X,
						gpumat::GpuMat **pAout,
						bool use_drop, double p)
{
	if(X.empty())
		return;

	if(use_drop)
		setDropout(p, 4);
	else
		clearDropout();

	gpumat::etypefunction func = gpumat::RELU;

	m_Xs.resize(m_conv.size());
	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].conv(X[i], m_Xs[i]);
	}

	gpumat::hconcat(m_Xs, m_Xout);

	gpumat::GpuMat *pA = &m_Xout;

	for(size_t i = 0; i < m_layers.size(); i++){
		gpumat::mlp& _mlp = m_mlp[i];

		if(i == m_layers.size() - 1)
			func = gpumat::SOFTMAX;

		_mlp.forward(pA, func);
		pA = &_mlp.A1;
	}

	*pAout = &(m_mlp.back().A1);

//	test_void(**pAout);
}

void gpu_train::pass(const std::vector<ct::Matf> &X, const ct::Matf &y)
{
	m_XsIn.resize(X.size());

	for(size_t i = 0; i < X.size(); ++i){
		gpumat::convert_to_gpu(X[i], m_XsIn[i]);
	}

	gpumat::convert_to_gpu(y, m_y_ind);

	pass();
}

void gpu_train::pass()
{
	gpumat::GpuMat *yp;
	forward(m_XsIn, &yp, true);

	////**********************

	gpumat::subIndOne(*yp, m_y_ind, m_td);

	///***********************

//	test_void(m_y_ind);
//	test_void(m_td);

	gpumat::GpuMat *pD = &m_td;

	for(int i = m_mlp.size() - 1; i > -1; --i){
		gpumat::mlp& mlp = m_mlp[i];

		mlp.backward(*pD);

		pD = &mlp.DltA0;
	}

	gpumat::hsplit(m_mlp.front().DltA0, m_conv.size(), m_splitD);

	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].backward(m_splitD[i]);
	}

	m_optim.pass(m_mlp);
}

std::vector<gpumat::tvconvnn> &gpu_train::cnv(int index)
{
	return m_conv[index].cnv();
}

void gpu_train::setDropout(float p, int layers)
{
	for(int i = 0; i < std::min(layers, (int)m_mlp.size() - 1); ++i){
		m_mlp[i].setDropout(true, p);
	}
}

void gpu_train::clearDropout()
{
	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].setDropout(false);
	}
}