#include "cifar_train.h"

////////////////////

/////////////////////

cifar_train::cifar_train()
{
	m_cifar = nullptr;
	m_init = false;
}

void cifar_train::setCifar(cifar_reader *val)
{
	m_cifar = val;
}

void cifar_train::setConvLayers(const std::vector<int> &layers,
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
}

void cifar_train::setMlpLayers(const std::vector<int> &layers)
{
	if(layers.empty())
		throw new std::invalid_argument("empty parameters");

	m_layers = layers;
}

void cifar_train::init()
{
	if(m_layers.empty() || m_cnvlayers.empty() || m_cnvweights.empty() || !m_cifar)
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
			ct::mlpf& mlp = m_mlp[i];
			int output = m_layers[i];

			mlp.init(input, output);

			input = output;
		}
	}

	m_optim.init(m_mlp);

	m_init = true;
}

void cifar_train::forward(const std::vector< ct::Matf > &X, ct::Matf &a_out,
						  bool use_drop, float p, bool use_gpu)
{
	if(X.empty())
		return;

	if(use_gpu && m_gpu_train.isInit()){
		m_gpu_train.forward(X, a_out, use_drop, p);
		return;
	}

	///********************

	{
		if(use_drop)
			setDropout(p, 4);
		else
			clearDropout();

		std::vector< ct::Matf > Xs_out;
		Xs_out.resize(m_conv.size());
		for(size_t i = 0; i < m_conv.size(); ++i){
			m_conv[i].conv(X[i], Xs_out[i]);
		}
		ct::hconcat(Xs_out, m_X_out);

		ct::Matf *pX = &m_X_out;

		for(size_t i = 0; i < m_mlp.size(); ++i){
			ct::mlpf& mlp = m_mlp[i];

			if(i < m_mlp.size() - 1){
				mlp.forward(pX, ct::RELU);
				pX = &mlp.A1;
			}else{
				mlp.forward(pX, ct::SOFTMAX);
			}
		}
		a_out = m_mlp.back().A1;
	}
}

void cifar_train::pass(int batch, bool use_gpu)
{
	if(!m_init || batch <= 0)
		throw new std::invalid_argument("not initialize");

	std::vector< ct::Matf > Xs;
	ct::Matf yp;
	ct::Matf y;

	m_cifar->getTrain(batch, Xs, y);

	if(use_gpu && m_gpu_train.isInit()){
		m_gpu_train.pass(Xs, y);
		return;
	}

	////**********************

	forward(Xs, yp, true);

	////**********************

	m_td = ct::subIndOne(yp, y);

	///***********************

	ct::Matf *pD = &m_td;

	for(int i = m_mlp.size() - 1; i > -1; --i){
		ct::mlpf& mlp = m_mlp[i];

		mlp.backward(*pD);

		pD = &mlp.DltA0;
	}

	ct::hsplit(m_mlp.front().DltA0, m_conv.size(), m_splitD);

	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].backward(m_splitD[i]);
	}

	m_optim.pass(m_mlp);
}

void cifar_train::getEstimage(int batch, double &accuracy, double &l2, bool use_gpu)
{
	if(!m_init || batch <= 0)
		throw new std::invalid_argument("not initialize");

	std::vector< ct::Matf > Xs;
	ct::Matf yp;
	ct::Matf y;

	m_cifar->getTrain(batch, Xs, y);

	if(use_gpu && m_gpu_train.isInit()){
		m_gpu_train.forward(Xs, yp);
		l2 = m_gpu_train.getL2(yp, y);
	}else{
		forward(Xs, yp);
		m_td = ct::subIndOne(yp,y);

		ct::Matf d = ct::elemwiseSqr(m_td);
		l2 = d.sum() / d.rows;
	}

	int count = 0;
	for(int i = 0; i < yp.rows; ++i){
		int id = yp.argmax(i, 1);
		if(y.at(i, 0) == id){
			count++;
		}
	}
	accuracy = (double)count / y.rows;
}

void cifar_train::setAlpha(double alpha)
{
	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].setAlpha(alpha);
	}
	m_optim.setAlpha(alpha);

	if(m_gpu_train.isInit()){
		m_gpu_train.setAlpha(alpha);
	}
}

uint cifar_train::iteration() const
{
	return m_optim.iteration();
}

uint cifar_train::iteration_gpu() const
{
	return m_gpu_train.iteration();
}

QVector< int > cifar_train::predict(double percent, int batch, bool use_gpu)
{
	QVector< int > pred;

	std::vector< ct::Matf > X;
	ct::Matf y;

	m_cifar->getTrainIt(percent, batch, X);

	forward(X, y, false, 0.92, use_gpu);

	pred.resize(y.rows);

	for(int i = 0; i < y.rows; ++i){
		pred[i] = y.argmax(i, 1);
	}
	return pred;
}

QVector<QVector<ct::Matf> > cifar_train::cnvW(int index, bool use_gpu)
{
	QVector< QVector < ct::Matf > > res;

	if(!use_gpu){
		std::vector<tvconvnnf> &cnv = m_conv[index]();
		res.resize(cnv.size());

		for(size_t i = 0; i < cnv.size(); ++i){
			for(size_t j = 0; j < cnv[i].size(); ++j){
				for(size_t k = 0; k < cnv[i][j].W.size(); ++k){
					res[i].push_back(cnv[i][j].W[k]);
				}
			}
		}

	}else{
		res.resize(m_gpu_train.cnv(index).size());
		std::vector< std::vector< gpumat::convnn > > &cnv = m_gpu_train.cnv(index);

		for(size_t i = 0; i < cnv.size(); ++i){
			for(size_t j = 0; j < cnv[i].size(); ++j){
				for(size_t k = 0; k < cnv[i][j].W.size(); ++k){
					ct::Matf Wf;
					gpumat::convert_to_mat(cnv[i][j].W[k], Wf);
					res[i].push_back(Wf);
				}
			}
		}
	}
	return res;
}

void cifar_train::init_gpu()
{
	if(m_gpu_train.isInit())
		return;

	m_gpu_train.setConvLayers(m_cnvlayers, m_cnvweights, m_szA0);
	m_gpu_train.setMlpLayers(m_layers);

	m_gpu_train.init();
}

void cifar_train::setDropout(float p, int layers)
{
	for(int i = 0; i < std::min(layers, (int)m_mlp.size() - 1); ++i){
		ct::mlpf& mlp = m_mlp[i];
		mlp.setDropout(true, p);
	}
}

void cifar_train::clearDropout()
{
	for(size_t i = 0; i < m_mlp.size(); ++i){
		ct::mlpf& mlp = m_mlp[i];
		mlp.setDropout(false);
	}
}
