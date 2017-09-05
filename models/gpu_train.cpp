#include "gpu_train.h"

#include <QDebug>

#include "matops.h"

#include "qt_work_mat.h"

const int channels = 3;

//////////////////////
/// \brief test_void
/// \param mat
///

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
	m_dropoutProb = 0.9;
	m_is_debug = false;
}

void gpu_train::setConvLayers(const std::vector< ct::ParamsCnv >& layers,
							  const ct::Size szA0)
{
	if(layers.empty())
		throw new std::invalid_argument("empty parameters");

	m_cnvlayers = layers;
	m_szA0 = szA0;

	m_init = false;
}

void gpu_train::setMlpLayers(const std::vector<ct::ParamsMlp> &layers)
{
	if(layers.empty())
		throw new std::invalid_argument("empty parameters");

	m_layers = layers;

	m_init = false;
}

void gpu_train::setDebug(bool val)
{
	m_is_debug = val;
}

void gpu_train::setLambda(double val)
{
	for(size_t i = 0; i < m_mlp.size(); ++i){
		gpumat::mlp& m = m_mlp[i];
		m.setLambda(val);
	}
}

void gpu_train::setAlpha(double alpha)
{
	m_optim.setAlpha(alpha);
}

void gpu_train::setAlphaCnv(double alpha)
{
	////////////////////
	/// need realization
	////////////////////

	m_cnv_optim.setAlpha(alpha);
}

void gpu_train::init()
{
	if(m_init)
		return;

	if(m_layers.empty() || m_cnvlayers.empty())
		throw new std::invalid_argument("empty arguments");

	//// 1

	{
		////////////////////
		/// need realization
		////////////////////
		m_conv.resize(m_cnvlayers.size());

		int input = ::channels;
		ct::Size sz = m_szA0;
		for(size_t i = 0; i < m_conv.size(); ++i){
			gpumat::convnn_gpu& cnv = m_conv[i];
			ct::ParamsCnv& params = m_cnvlayers[i];
			ct::Size szW(params.size_w, params.size_w);
			cnv.init(sz, input, params.stride, params.count, szW, gpumat::LEAKYRELU, params.pooling, true, i != 0);
			cnv.setDropout(params.prob);
			cnv.setLambda(params.lambda_l2);
			input = params.count;
			sz = cnv.szOut();
		}
	}

	//// 2

	{
		m_mlp.resize(m_layers.size());

		int input = m_conv.back().outputFeatures();

		qDebug("MLP: input features = %d", input);

		for(size_t i = 0; i < m_mlp.size(); ++i){
			gpumat::mlp& mlp = m_mlp[i];
			ct::ParamsMlp &params = m_layers[i];
			int output = params.count;

			mlp.init(input, output, gpumat::GPU_FLOAT, i != m_mlp.size() - 1? gpumat::LEAKYRELU : gpumat::SOFTMAX);
			mlp.setDropout(params.prob);
			mlp.setLambda(params.lambda_l2);

			input = output;
		}
	}

	m_optim.init(m_mlp);
	m_cnv_optim.init(m_conv);

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

	ct::Matf red;
	gpumat::convert_to_mat(m_tsub, red);
//	gpumat::reduce(m_tsub, m_red);


	double l2 = red.sum();

	return l2 / y.rows;
}

void gpu_train::forward(const std::vector<ct::Matf> &X, ct::Matf &a_out, bool use_drop, bool use_ret)
{
	if(X.empty())
		return;

	m_XsIn.resize(X.size());

	for(size_t i = 0; i < X.size(); ++i){
		gpumat::convert_to_gpu(X[i], m_XsIn[i]);
	}

	gpumat::GpuMat *retA = nullptr;

	forward(m_XsIn, &retA, use_drop);

	if(use_ret && retA){
		gpumat::convert_to_mat(*retA, a_out);
	}
}

void gpu_train::forward(const std::vector<gpumat::GpuMat> &X,
						gpumat::GpuMat **pAout,
						bool use_drop)
{
	if(X.empty())
		return;

	if(use_drop)
		setDropout();
	else
		clearDropout();

	gpumat::etypefunction func = gpumat::RELU;

////////////////////
/// need realization
////////////////////
	if(use_drop)
		setDropout();
	else
		clearDropout();

	std::vector< gpumat::GpuMat > *pvX = (std::vector< gpumat::GpuMat >*)&X;

	for(int i = 0; i < (int)m_conv.size(); ++i){
		gpumat::convnn_gpu& cnv = m_conv[i];
		cnv.forward(pvX);
		pvX = &cnv.XOut();
	}

	gpumat::vec2mat(m_conv.back().XOut(), m_Xout);


	if(m_is_debug){
		ct::Matf mat, mn, _std;
		gpumat::convert_to_mat(m_Xout, mat);
		ct::get_mean(mat, mn);
		ct::get_std(mat, mn, _std);
		ct::save_mat(mn, "mean.txt");
		ct::save_mat(_std, "std.txt");
	}

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

	for(int i = (int)m_mlp.size() - 1; i > -1; --i){
		gpumat::mlp& mlp = m_mlp[i];

		mlp.backward(*pD);

		if(m_is_debug){
			std::stringstream ss;
			ss << "mlp_W" << i << ".txt";
			gpumat::save_gmat(mlp.W, ss.str());
			ss.str("");
			ss << "mlp_B" << i << ".txt";
			gpumat::save_gmat(mlp.B, ss.str());
		}

		pD = &mlp.DltA0;
	}

	if(m_is_debug){
		gpumat::save_gmat(m_mlp[0].DltA0, "mlp0.dlt.txt");
	}

	gpumat::mat2vec(m_mlp[0].DltA0, m_conv.back().szK, m_splitD);

	std::vector< gpumat::GpuMat >* pX = &m_splitD;

	for(int i = (int)m_conv.size() - 1; i > -1; --i){
		gpumat::convnn_gpu& cnv = m_conv[i];

		cnv.backward(*pX, i == 0);

		if(m_is_debug){
			std::stringstream ss;
			ss << "cnv_W" << i << ".txt";
			gpumat::save_gmat(cnv.W, ss.str());
			ss.str("");
			ss << "cnv_B" << i << ".txt";
			gpumat::save_gmat(cnv.B, ss.str());
		}

		pX = &cnv.Dlt;
	}

	m_cnv_optim.pass(m_conv);
	m_optim.pass(m_mlp);
}

template< typename T >
inline void write_vector(std::fstream& fs, const std::vector<T> &vec)
{
	int tmp = (int)vec.size();
	fs.write((char*)&tmp, sizeof(tmp));
	if(tmp){
		for(size_t i = 0; i < vec.size(); ++i){
			vec[i].write(fs);
		}
	}
}

template< typename T >
inline void read_vector(std::fstream& fs, std::vector<T>  &vec)
{
	int tmp;
	fs.read((char*)&tmp, sizeof(tmp));
	if(tmp){
		for(size_t i = 0; i < vec.size(); ++i){
			vec[i].read(fs);
		}
	}
}

bool gpu_train::loadFromFile(const std::string &fn)
{
	std::fstream fs;
	fs.open(fn, std::ios_base::in | std::ios_base::binary);

	if(!fs.is_open()){
		qDebug("File %s not open", fn.c_str());
		return false;
	}

	read_vector(fs, m_cnvlayers);
	read_vector(fs, m_layers);

	fs.read((char*)&m_szA0, sizeof(m_szA0));

	setConvLayers(m_cnvlayers, m_szA0);

	init();

	////////////////////
	/// need realization
	////////////////////
	for(size_t i = 0; i < m_conv.size(); ++i){
		gpumat::convnn_gpu& cnv = m_conv[i];
		cnv.read(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].read(fs);
	}

	return true;
}

void gpu_train::saveToFile(const std::string &fn)
{
	std::fstream fs;
	fs.open(fn, std::ios_base::out | std::ios_base::binary);

	if(!fs.is_open()){
		qDebug("File %s not open", fn.c_str());
		return;
	}

	write_vector(fs, m_cnvlayers);
	write_vector(fs, m_layers);

	fs.write((char*)&m_szA0, sizeof(m_szA0));

	////////////////////
	/// need realization
	////////////////////

	for(size_t i = 0; i < m_conv.size(); ++i){
		gpumat::convnn_gpu& cnv = m_conv[i];
		cnv.write(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].write(fs);
	}

	//	qt_work_mat::q_save_mat(m_mlp.back().W, "W0back.txt");
}

uint gpu_train::outputFeatures() const
{
	const gpumat::convnn_gpu& cnv = m_conv.back();
	return cnv.outputFeatures();
}

std::vector<gpumat::convnn_gpu> &gpu_train::conv()
{
	return m_conv;
}

void gpu_train::setDropoutProb(double val)
{
	m_dropoutProb = val;
	for(int i = 0; i < (int)m_mlp.size() - 2; ++i){
		gpumat::mlp& m = m_mlp[i];
		m.setDropout(val);
	}
}

double gpu_train::dropoutProb() const
{
	return m_dropoutProb;
}

void gpu_train::save_weights()
{
	for(size_t i = 0; i < m_conv.size(); ++i){
		QString name = "cnvW_" + QString::number(i) + ".txt";
		qt_work_mat::q_save_mat(m_conv[i].W, name);
		name = "cnvB_" + QString::number(i) + ".txt";
		qt_work_mat::q_save_mat(m_conv[i].B, name);
	}
	for(size_t i = 0; i < m_mlp.size(); ++i){
		QString name = "mlpW_" + QString::number(i) + ".txt";
		qt_work_mat::q_save_mat(m_mlp[i].W, name);
		name = "mlpB_" + QString::number(i) + ".txt";
		qt_work_mat::q_save_mat(m_mlp[i].B, name);
	}
}

void gpu_train::setDropout()
{
	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].setDropout(true);
	}
	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].setDropout(true);
	}
}

void gpu_train::clearDropout()
{
	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].setDropout(false);
	}
	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].setDropout(false);
	}
}
