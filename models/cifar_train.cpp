#include "cifar_train.h"

#include <QMap>
////////////////////

template< typename T >
void translate(int x, int y, int w, int h, T *X)
{
	std::vector<T>d;
	d.resize(w * h, 0);

#pragma omp parallel for
	for(int i = 0; i < h; i++){
		int newi = i + x;
		if(newi >= 0 && newi < h){
			for(int j = 0; j < w; j++){
				int newj = j + y;
				if(newj >= 0 && newj < w){
					d[newi * w + newj] = X[i * w + j];
				}
			}
		}
	}
	for(size_t i = 0; i < d.size(); i++){
		X[i] = d[i];
	}
}

template< typename T >
void rotate_data(int w, int h, T angle, T *X)
{
	T cw = w / 2;
	T ch = h / 2;

	std::vector<T> d;
	d.resize(w * h, 0);

	for(int y = 0; y < h; y++){
		for(int x = 0; x < w; x++){
			T x1 = x - cw;
			T y1 = y - ch;

			T nx = x1 * cos(angle) + y1 * sin(angle);
			T ny = -x1 * sin(angle) + y1 * cos(angle);
			nx += cw; ny += ch;
			int ix = nx, iy = ny;
			if(ix >= 0 && ix < w && iy >= 0 && iy < h){
				T c = X[y * w + x];
				d[iy * w + ix] = c;
			}
		}
	}
	for(size_t i = 0; i < d.size(); i++){
		X[i] = d[i];
	}
}

/////////////////////

cifar_train::cifar_train()
{
	m_cifar = nullptr;
	m_init = false;
	m_rand_data = ct::Vec3f(3, 3);
	m_use_rand_data = false;
}

void cifar_train::setCifar(cifar_reader *val)
{
	m_cifar = val;
}

void cifar_train::setConvLayers(const std::vector<int> &layers,
								std::vector<int> weight_sizes,
								const ct::Size szA0,
								std::vector< char > *pooling)
{
	if(layers.empty() || weight_sizes.empty())
		throw new std::invalid_argument("empty parameters");

	if(pooling)
		m_cnvpooling = *pooling;
	m_cnvlayers = layers;
	m_cnvweights = weight_sizes;
	m_szA0 = szA0;

	m_conv.resize(3);
	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].setConvLayers(layers, weight_sizes, szA0, pooling);
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

void cifar_train::setRandData(float offset, float angle)
{
	m_rand_data[0] = offset;
	m_rand_data[1] = angle;
}

void cifar_train::setUseRandData(bool val)
{
	m_use_rand_data = val;
}


void cifar_train::randValues(size_t count, std::vector<ct::Vec3f> &vals, float offset, float angle)
{
	std::uniform_int_distribution<int> udtr(-offset, offset);
	std::uniform_real_distribution<float> uar(-angle, angle);

	vals.resize(count);

	for(size_t i = 0; i < vals.size(); ++i){
		int x = udtr(ct::generator);
		int y = udtr(ct::generator);
		float ang = uar(ct::generator);
		ang = ct::angle2rad(ang);
		vals[i] = ct::Vec3f(x, y, ang);
	}
}

void cifar_train::randX(ct::Matf &X, std::vector<ct::Vec3f> &vals)
{
	if(X.empty() || X.rows != vals.size())
		return;
#if 1

#pragma omp parallel for
	for(int i = 0; i < X.rows; i++){
		float *Xi = &X.at(i, 0);

		float x = vals[i][0];
		float y = vals[i][1];
		float ang = vals[i][2];

		rotate_data<float>(cifar_reader::WidthIM, cifar_reader::HeightIM, ang, Xi);
		translate<float>(x, y, cifar_reader::WidthIM, cifar_reader::HeightIM, Xi);
	}
#endif
}

void cifar_train::pass(int batch, bool use_gpu)
{
	if(!m_init || batch <= 0)
		throw new std::invalid_argument("not initialize");

	std::vector< ct::Matf > Xs;
	ct::Matf yp;
	ct::Matf y;

	m_cifar->getTrain(batch, Xs, y);

	if(m_use_rand_data){
		randValues(y.rows, m_vals, m_rand_data[0], m_rand_data[1]);

		for(size_t i = 0; i < Xs.size(); ++i){
			randX(Xs[i], m_vals);
		}
	}

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


void cifar_train::getEstimate(const std::vector<ct::Matf> &Xs, ct::Matf &y,
							  uint &right, double &l2, bool use_gpu)
{
	ct::Matf yp;

	if(Xs.empty() || Xs[0].empty() || y.empty())
		return;

	if(use_gpu && m_gpu_train.isInit()){
		sliceEstimage(Xs, y, right, l2);
	}else{
		forward(Xs, yp);

		m_td = ct::subIndOne(yp, y);

		ct::Matf d = ct::elemwiseSqr(m_td);
		l2 = d.sum() / d.rows;

		right = getRight(y, yp);
	}

}

void getSlice(const std::vector<ct::Matf> &Xs, int first, int last, std::vector<ct::Matf> &Xsi)
{
	Xsi.resize(Xs.size());

	for(size_t i = 0; i < Xs.size(); ++i){
		const ct::Matf& mat = Xs[i];
		ct::Matf &mati = Xsi[i];
		mati.setSize(last - first, mat.cols);
		float *dM = mat.ptr();
		float *dMi = mati.ptr();
#pragma omp parallel for
		for(int l = 0; l < mati.rows; ++l){
			int j = first + l;
			for(int k = 0; k < mati.cols; ++k){
				dMi[l * mati.cols + k] = dM[j * mat.cols + k];
			}
		}
	}
}

void getSlice(const ct::Matf &Xs, int first, int last, ct::Matf &Xsi)
{
	Xsi.setSize(last - first, Xs.cols);
	float *dM = Xs.ptr();
	float *dMi = Xsi.ptr();
#pragma omp parallel for
	for(int l = 0; l < Xsi.rows; ++l){
		int j = first + l;
		for(int k = 0; k < Xsi.cols; ++k){
			dMi[l * Xsi.cols + k] = dM[j * Xs.cols + k];
		}
	}
}


void cifar_train::sliceEstimage(const std::vector<ct::Matf> &Xs, ct::Matf &y, uint &right, double &l2)
{
	int batch = 100;

	int cnt = y.rows / batch;
	if(!cnt)cnt = 1;

	l2 = 0;
	right = 0;

	for (int i = 0; i < cnt; ++i){
		int size = std::min(y.rows - batch * i, batch);
		std::vector< ct::Matf > Xsi;
		ct::Matf ysi, ypi;
		getSlice(Xs, i * batch, i * batch + size, Xsi);
		getSlice(y, i * batch, i * batch + size, ysi);

		m_gpu_train.forward(Xsi, ypi);

		l2 += m_gpu_train.getL2(ypi, ysi);
		right += getRight(ysi, ypi);
	}
	l2 /= cnt;
}

int cifar_train::getRight(const ct::Matf &y, const ct::Matf &yp)
{
	int count = 0;
	for(int i = 0; i < yp.rows; ++i){
		int idp = yp.argmax(i, 1);
		int idy = y.at(i, 0);
		ct::Vec2i vec = m_statistics[idy];
		if(idy == idp){
			count++;
			vec[0]++;
		}
		vec[1]++;
		m_statistics[idy] = vec;
	}
	return count;
}

void cifar_train::getEstimate(int batch, double &accuracy, double &l2, bool use_gpu)
{
	if(!m_init || batch <= 0)
		throw new std::invalid_argument("not initialize");

	std::vector< ct::Matf > Xs;
	ct::Matf y;

	m_cifar->getTrain(batch, Xs, y);

	m_statistics.clear();

	uint right;
	getEstimate(Xs, y, right, l2, use_gpu);

	accuracy = (double)right / y.rows;
}

void cifar_train::getEstimateTest(double &accuracy, double &l2, bool use_gpu)
{
	std::vector< ct::Matf > Xs;
	ct::Matf y;

	uint batch = 300, ind = 0, right = 0, right_all = 0, count_all = 0;
	double l2i;

	uint size = m_cifar->count_test();

	m_statistics.clear();

	while(ind < size){
		batch = m_cifar->getTest(ind, batch, Xs, y);

		getEstimate(Xs, y, right, l2i, use_gpu);
		l2 += l2i;
		right_all += right;

		count_all++;

		ind += batch;
		qDebug("test pass: pos=%d, batch=%d, count=%d", ind, batch, size);
	}
	accuracy = (double)right_all / m_cifar->count_test();
	l2 /= count_all;
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

QVector< int > cifar_train::predict(const QVector< TData >& data, bool use_gpu)
{
	QVector< int > pred;

	std::vector< ct::Matf > X;
	ct::Matf y;

	int batch = 100;

	pred.resize(data.size());

	for(int i = 0; i < data.size(); i += batch){

		int cnt = std::min(data.size() - i, batch);

		m_cifar->convToXy(data, i, i + cnt, X);

		forward(X, y, false, 0.92, use_gpu);

		for(int j = 0; j < y.rows; ++j){
			pred[i + j] = y.argmax(j, 1);
		}

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

	m_gpu_train.setConvLayers(m_cnvlayers, m_cnvweights, m_szA0, &m_cnvpooling);
	m_gpu_train.setMlpLayers(m_layers);

	m_gpu_train.init();
}

bool cifar_train::loadFromFile(const QString &fn, bool gpu)
{
	if(gpu){
		return m_gpu_train.loadFromFile((fn + ".gpu").toStdString());
	}

	std::fstream fs;
	fs.open(fn.toStdString(), std::ios_base::in | std::ios_base::binary);

	if(!fs.is_open()){
		qDebug("File %s not open", fn.toLatin1().data());
		return false;
	}

	int tmp;

	fs.read((char*)&tmp, sizeof(tmp));
	m_cnvlayers.resize(tmp);
	fs.read((char*)&m_cnvlayers[0], m_cnvlayers.size() * sizeof(decltype(m_cnvlayers)::value_type));

	fs.read((char*)&tmp, sizeof(tmp));
	m_cnvweights.resize(tmp);
	fs.read((char*)&m_cnvweights[0], m_cnvweights.size() * sizeof(decltype(m_cnvweights)::size_type));

	fs.read((char*)&tmp, sizeof(tmp));
	m_layers.resize(tmp);
	fs.read((char*)&m_layers[0], m_layers.size() * sizeof(decltype(m_layers)::size_type));

	fs.read((char*)&m_szA0, sizeof(m_szA0));

	setConvLayers(m_cnvlayers, m_cnvweights, m_szA0);

	init();

	for(size_t i = 0; i < m_conv.size(); ++i){
		convnn::ConvNN &cnv = m_conv[i];
		cnv.read(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].read(fs);
	}
	return true;
}

void cifar_train::saveToFile(const QString &fn, bool gpu)
{
	if(gpu){
		m_gpu_train.saveToFile((fn + ".gpu").toStdString());
		return;
	}

	std::fstream fs;
	fs.open(fn.toStdString(), std::ios_base::out | std::ios_base::binary);

	if(!fs.is_open()){
		qDebug("File %s not open", fn.toLatin1().data());
		return;
	}

	int tmp;

	tmp = m_cnvlayers.size();
	fs.write((char*)&tmp, sizeof(tmp));
	fs.write((char*)&m_cnvlayers[0], m_cnvlayers.size() * sizeof(decltype(m_cnvlayers)::value_type));

	tmp = m_cnvweights.size();
	fs.write((char*)&tmp, sizeof(tmp));
	fs.write((char*)&m_cnvweights[0], m_cnvweights.size() * sizeof(decltype(m_cnvweights)::size_type));

	tmp = m_layers.size();
	fs.write((char*)&tmp, sizeof(tmp));
	fs.write((char*)&m_layers[0], m_layers.size() * sizeof(decltype(m_layers)::size_type));

	fs.write((char*)&m_szA0, sizeof(m_szA0));

	for(size_t i = 0; i < m_conv.size(); ++i){
		convnn::ConvNN &cnv = m_conv[i];
		cnv.write(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].write(fs);
	}

}

ct::Vec2i cifar_train::statistics(int val) const
{
	return m_statistics[val];
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
