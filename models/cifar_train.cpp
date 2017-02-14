#include "cifar_train.h"


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
						  bool use_drop, float p)
{
	if(X.empty())
		return;

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

void cifar_train::pass(int batch)
{
	if(!m_init || batch <= 0)
		throw new std::invalid_argument("not initialize");

	std::vector< ct::Matf > Xs;
	ct::Matf y, yp;

	m_cifar->getTrain(batch, Xs, y);

	////**********************

	forward(Xs, yp, true);

	////**********************

	m_td = yp - y;

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

void cifar_train::getEstimage(int batch, double &accuracy, double &l2)
{
	if(!m_init || batch <= 0)
		throw new std::invalid_argument("not initialize");

	std::vector< ct::Matf > Xs;
	ct::Matf y, yp;

	m_cifar->getTrain(batch, Xs, y);

	forward(Xs, yp);

	m_td = yp - y;

	ct::Matf d = ct::elemwiseSqr(m_td);
	l2 = d.sum() / d.rows;

	int count = 0;
	for(int i = 0; i < yp.rows; ++i){
		int id = yp.argmax(i, 1);
		if(y.at(i, id) == 1){
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
}

uint cifar_train::iteration() const
{
	return m_optim.iteration();
}

QVector< int > cifar_train::predict(double percent, int batch)
{
	QVector< int > pred;

	std::vector< ct::Matf > X;
	ct::Matf y;

	m_cifar->getTrainIt(percent, batch, X);

	forward(X, y);

	pred.resize(y.rows);

	for(int i = 0; i < y.rows; ++i){
		pred[i] = y.argmax(i, 1);
	}
	return pred;
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
