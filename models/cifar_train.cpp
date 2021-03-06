#include "cifar_train.h"
#include <algorithm>

#include <omp.h>

#include <QMap>
////////////////////

const int channels = 3;

const int count_classes = 10;

template< typename T >
void flip(int w, int h, T *X, std::vector<T> &d)
{
	if((int)d.size() != w * h)
		d.resize(w * h);
	std::fill(d.begin(), d.end(), 0);

//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			int newj = w - j - 1;
			d[i * w + newj] = X[i * w + j];
		}
	}
	for(size_t i = 0; i < d.size(); i++){
		X[i] = d[i];
	}
}

template< typename T >
void exchange(int w, int h, T *X1, T *X2)
{
	std::swap_ranges(X1, X1 + w * h, X2);
}

template< typename T >
void crop(int w, int h, int new_w, int new_h, int offx, int offy, T *X, std::vector<T> &d)
{
	if(new_w > w || new_h > h)
		return;

	if((int)d.size() != w * h)
		d.resize(w * h);
	std::fill(d.begin(), d.end(), 0);

	int xc = w / 2, yc = h / 2;
//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = yc - new_h/2; i < yc + new_h/2; i++){
		int y = offy + i;
		if(y >= 0 && y < h){
			for(int j = xc - new_w/2; j < xc + new_w/2; j++){
				int x = offx + j;

				if(x >= 0 && x < w){
					d[y * w + x] = X[y * w + x];
				}
			}
		}
	}
	for(size_t i = 0; i < d.size(); i++){
		X[i] = d[i];
	}
}

template< typename T >
void translate(int x, int y, int w, int h, T *X, std::vector<T> &d)
{
	if((int)d.size() != w * h)
		d.resize(w * h);
	std::fill(d.begin(), d.end(), 0);

//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#endif
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
void rotate_data(int w, int h, T angle, T *X, std::vector<T> &d)
{
	int cw = w / 2;
	int ch = h / 2;

	if((int)d.size() != w * h)
		d.resize(w * h);

	std::fill(d.begin(), d.end(), 0);

//	T delta = 0.4 * angle * angle;

#ifdef __GNUC__
#pragma omp simd
#endif
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


template< typename T >
inline T __tr(T val)
{
	if(val > 1)
		val = 1;
	if(val < 0)
		val = 0;
	return val;
}
template <typename T>
void change_brightness(ct::Mat_<T>& mat, T val)
{
	T* dM = mat.ptr();
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < mat.total(); ++i){
		dM[i] = __tr(dM[i] * val);
	}
}

template <typename T>
void change_contrast(int w, int h, T* r, T* g, T* b, T C)
{
	T F = (259. * (C + 255.)) / (255. * (259. - C));

#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < w * h; ++i){
			r[i] = __tr(F * (r[i] - 0.5) + 0.5);
			g[i] = __tr(F * (g[i] - 0.5) + 0.5);
			b[i] = __tr(F * (b[i] - 0.5) + 0.5);
	}
}

/////////////////////

cifar_train::cifar_train()
{
	m_cifar = nullptr;
	m_init = false;
	m_rand_data = ct::Vec3f(3, 3);
	m_use_rand_data = false;
	m_dropoutProb = 0.9;

	m_statistics.resize(count_classes * count_classes);
}

void cifar_train::setCifar(cifar_reader *val)
{
	m_cifar = val;
}

void cifar_train::setConvLayers(const std::vector<ct::ParamsCnv> &layers,
								const ct::Size szA0)
{
	if(layers.empty())
		throw new std::invalid_argument("empty parameters");

	m_cnvlayers = layers;
	m_szA0 = szA0;
}

void cifar_train::setMlpLayers(const std::vector<ct::ParamsMlp> &layers)
{
	if(layers.empty())
		throw new std::invalid_argument("empty parameters");

	m_layers = layers;
}

void cifar_train::init()
{
	if(m_layers.empty() || m_cnvlayers.empty() || !m_cifar)
		throw new std::invalid_argument("empty arguments");

	//// 1

	{
		m_conv.resize(m_cnvlayers.size());

		int input = ::channels;
		ct::Size sz = m_szA0;
		for(size_t i = 0; i < m_conv.size(); ++i){
			conv2::convnn<float>& cnv = m_conv[i];
			ct::ParamsCnv& params = m_cnvlayers[i];

			ct::Size szW(params.size_w, params.size_w);

			cnv.init(sz, input, params.stride, params.count, szW, ct::LEAKYRELU, params.pooling, true, i != 0, params.same);
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
			ct::mlpf& mlp = m_mlp[i];
			ct::ParamsMlp& params = m_layers[i];

			int output = params.count;

			mlp.init(input, output, i != m_mlp.size() - 1? ct::LEAKYRELU : ct::SOFTMAX);
			mlp.setDropout((float)params.prob);
			mlp.setLambda(params.lambda_l2);

			input = output;
		}
	}

	m_optim.init(m_mlp);
	m_cnv_optim.init(m_conv);

	m_init = true;
}

void cifar_train::forward(const std::vector< ct::Matf > &X, ct::Matf &a_out,
						  bool use_drop, bool use_gpu)
{
	if(X.empty())
		return;

	if(use_gpu && m_gpu_train.isInit()){
		m_gpu_train.forward(X, a_out, use_drop);
		return;
	}

	///********************

	{
		if(use_drop)
			setDropout();
		else
			clearDropout();

		std::vector< ct::Matf > *pvX = (std::vector< ct::Matf >*)&X;

		for(size_t i = 0; i < m_conv.size(); ++i){
			conv2::convnn<float>& cnv = m_conv[i];
			cnv.forward(pvX);
			pvX = &cnv.XOut();
		}

		conv2::vec2mat(m_conv.back().XOut(), m_X_out);

		ct::Matf *pX = &m_X_out;

		for(size_t i = 0; i < m_mlp.size(); ++i){
			ct::mlpf& mlp = m_mlp[i];

			mlp.forward(pX, ct::RELU);
			pX = &mlp.A1;
		}
		a_out = m_mlp.back().A1;
	}
}

void cifar_train::setRandData(float offset, float angle, float br)
{
	m_rand_data[0] = offset;
	m_rand_data[1] = angle;
	m_rand_data[2] = br;
}

void cifar_train::setUseRandData(bool val)
{
	m_use_rand_data = val;
}


void cifar_train::randValues(size_t count, std::vector<ct::Vec4f> &vals, float offset, float angle, float brightness)
{
	vals.resize(count);
	for(size_t i = 0; i < vals.size(); ++i){
		ct::Vec4f& v = vals[i];
		v = ct::Vec4f::zeros();
	}

	if(offset > 0){
		std::uniform_int_distribution<int> udtr(-offset, offset);
		for(size_t i = 0; i < vals.size(); ++i){
			int x = udtr(ct::generator);
			int y = udtr(ct::generator);
			ct::Vec4f& v = vals[i];
			v[0] = x;
			v[1] = y;
		}
	}

	if(angle > 0){
		angle = ct::angle2rad(angle);
		std::uniform_real_distribution<float> uar(-angle, angle);
		for(size_t i = 0; i < vals.size(); ++i){
			float ang = uar(ct::generator);
			vals[i][2] = ang;
		}
	}
	if(brightness > 0){
		std::uniform_real_distribution<float> nbr(1 - brightness, 1);
		for(size_t i = 0; i < vals.size(); ++i){
			float br = nbr(ct::generator);
			vals[i][3] = br;
		}
	}
}

#include <QImage>

void saveIm(float* dx1, float *dx2, float *dx3, int width, int height, const QString& name)
{
	QImage im(width, height, QImage::Format_ARGB32);

	for(int y = 0; y < height; ++y){
		QRgb *sc = (QRgb*)im.scanLine(y);
		for(int x = 0; x < width; ++x){
			float c1 = dx1[y * width + x];
			float c2 = dx2[y * width + x];
			float c3 = dx3[y * width + x];
			c1 = 255. * (c1);
			c2 = 255. * (c2);
			c3 = 255. * (c3);
			uchar uc1 = c1;
			uchar uc2 = c2;
			uchar uc3 = c3;
			sc[x] = qRgb(uc1, uc2, uc3);
		}
	}
	im.save(name);
}

void cifar_train::randX(std::vector< ct::Matf > &X, std::vector<ct::Vec4f> &vals)
{
	if(X.empty() || X.size() != vals.size())
		return;
#if 1

	int area = cifar_reader::WidthIM * cifar_reader::HeightIM;

	std::binomial_distribution<int> ufl(1, 0.5);
	std::uniform_int_distribution<int> udtr(-m_rand_data[0], m_rand_data[0]);
	std::uniform_real_distribution<float> uctr(-100, 100);

	int max_threads = omp_get_num_procs();

	omp_set_num_threads(max_threads * 2);

	std::vector< std::vector< float > > ds;
	ds.resize(max_threads * 2);

	int w = cifar_reader::WidthIM, h = cifar_reader::HeightIM;

#pragma omp parallel for
	for(int i = 0; i < (int)X.size(); i++){
		float *dX = X[i].ptr();

		float *dX1 = &dX[0 * area];
		float *dX2 = &dX[1 * area];
		float *dX3 = &dX[2 * area];

//		float x = vals[i][0];
//		float y = vals[i][1];
		float ang = vals[i][2];
		float br = vals[i][3];

		int _num = omp_get_thread_num();
		float x1 = udtr(ct::generator);
		float y1 = udtr(ct::generator);

		std::vector< float >& d = ds[_num];

		int fl = ufl(ct::generator);
//		int fl1 = ufl(ct::generator);
//		int fl2 = ufl(ct::generator);
//		int fl3 = ufl(ct::generator);
		float ctr = uctr(ct::generator);

		change_contrast<float>(w, h, dX1, dX2, dX3, ctr);

//		if(fl1 > 0){
//			crop<float>(cifar_reader::WidthIM, cifar_reader::HeightIM, 24, 24, x, y, dX1, d);
//			crop<float>(cifar_reader::WidthIM, cifar_reader::HeightIM, 24, 24, x, y, dX2, d);
//			crop<float>(cifar_reader::WidthIM, cifar_reader::HeightIM, 24, 24, x, y, dX3, d);
//		}

		if(fl > 0){
			flip<float>(w, h, dX1, d);
			flip<float>(w, h, dX2, d);
			flip<float>(w, h, dX3, d);
		}

//		if(fl2 > 0){
//			exchange<float>(w, h, dX1, dX2);
//		}
//		if(fl3 > 0){
//			exchange<float>(w, h, dX1, dX3);
//		}

		if(ang){
			rotate_data<float>(w, h, ang, dX1, d);
			rotate_data<float>(w, h, ang, dX2, d);
			rotate_data<float>(w, h, ang, dX3, d);
		}

		if(x1 && y1){
			translate<float>(x1, y1, w, h, dX1, d);
			translate<float>(x1, y1, w, h, dX2, d);
			translate<float>(x1, y1, w, h, dX3, d);
		}

		if(br){
			change_brightness(X[i], br);
		}

//		QString name = "data/image_" + QString::number(i) + ".bmp";
//		saveIm(dX1, dX2, dX3, w, h,
//			   "data/image_" + QString::number(i) + ".bmp");
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

	m_cifar->getTrain2(batch, Xs, y);

	if(m_use_rand_data){
		randValues(y.rows, m_vals, m_rand_data[0], m_rand_data[1], m_rand_data[2]);

		randX(Xs, m_vals);
	}

	if(use_gpu && m_gpu_train.isInit()){
		m_gpu_train.pass(Xs, y);
		return;
	}

	////**********************

	forward(Xs, yp, true, m_dropoutProb);

	////**********************

	m_td = ct::subIndOne(yp, y);

	///***********************

	ct::Matf *pD = &m_td;

	for(int i = (int)m_mlp.size() - 1; i > -1; --i){
		ct::mlpf& mlp = m_mlp[i];

		mlp.backward(*pD);

		pD = &mlp.DltA0;
	}

	std::vector< ct::Matf > vm, *pvm;

	conv2::mat2vec(m_mlp.front().DltA0, m_conv.back().szK, vm);
	pvm = &vm;

	for(int i = (int)m_conv.size() - 1; i > -1; --i){
		conv2::convnn<float>& cnv = m_conv[i];

		cnv.backward(*pvm, i == 0);
		pvm = &cnv.Dlt;
	}

	m_optim.pass(m_mlp);
	m_cnv_optim.pass(m_conv);
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
	Xsi.resize(last - first);

	for(size_t i = 0; i < Xsi.size(); ++i){
		Xsi[i] = Xs[first + i];
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
	int batch = 50;

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
		ct::Vec2i& v1 = m_statistics[idy * count_classes + idp];
		ct::Vec2i& v2 = m_statistics[idy * count_classes + idy];
		if(idy == idp){
			count++;
		}
		v1[0]++;
		v2[1]++;
	}
	return count;
}

void cifar_train::getEstimate(int batch, double &accuracy, double &l2, bool use_gpu)
{
	if(!m_init || batch <= 0)
		throw new std::invalid_argument("not initialize");

	std::vector< ct::Matf > Xs;
	ct::Matf y;

	m_cifar->getTrain2(batch, Xs, y);

	std::fill(m_statistics.data(), m_statistics.data() + m_statistics.size(), 0);

	uint right;
	getEstimate(Xs, y, right, l2, use_gpu);

	accuracy = (double)right / y.rows;
}

void cifar_train::getEstimateTest(int batch, double &accuracy, double &l2, bool use_gpu)
{
	if(!m_init || batch <= 0)
		throw new std::invalid_argument("not initialize");

	std::vector< ct::Matf > Xs;
	ct::Matf y;

	m_cifar->getTest2(batch, Xs, y);

	std::fill(m_statistics.data(), m_statistics.data() + m_statistics.size(), 0);

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

	std::fill(m_statistics.data(), m_statistics.data() + m_statistics.size(), 0);

	while(ind < size){
		batch = m_cifar->getTest2(ind, batch, Xs, y);

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
	m_optim.setAlpha(alpha);

	if(m_gpu_train.isInit()){
		m_gpu_train.setAlpha(alpha);
	}
}

void cifar_train::setAlphaCnv(double val)
{
	m_cnv_optim.setAlpha(val);
	if(m_gpu_train.isInit()){
		m_gpu_train.setAlphaCnv(val);
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


uint cifar_train::inputToMlp(bool use_gpu) const
{
	if(use_gpu)
		return m_gpu_train.outputFeatures();
	else
		return m_conv.back().outputFeatures();
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

		m_cifar->convToXy2(data, i, i + cnt, X);

		forward(X, y, false, use_gpu);

		for(int j = 0; j < y.rows; ++j){
			pred[i + j] = y.argmax(j, 1);
		}

	}

	return pred;
}

ct::Matf cifar_train::cnvW(int index, bool use_gpu)
{
	if(index >= (int)m_cnvlayers.size())
		index = (int)m_cnvlayers.size() - 1;

	if(!use_gpu){
		return m_conv[index].W;
	}else{
		gpumat::GpuMat& gW = m_gpu_train.conv()[index].W;
		ct::Matf W;
		gpumat::convert_to_mat(gW, W);
		return W;
//		res.resize(m_gpu_train.cnv(index).size());
//		std::vector< std::vector< gpumat::convnn > > &cnv = m_gpu_train.cnv(index);

//		for(size_t i = 0; i < cnv.size(); ++i){
//			for(size_t j = 0; j < cnv[i].size(); ++j){
//				for(size_t k = 0; k < cnv[i][j].W.size(); ++k){
//					ct::Matf Wf;
//					gpumat::convert_to_mat(cnv[i][j].W[k], Wf);
//					res[i].push_back(Wf);
//				}
//			}
//		}
	}
	return ct::Matf();
}

ct::Size &cifar_train::szW(int index, bool use_gpu)
{
	if(index >= (int)m_cnvlayers.size())
		index = (int)m_cnvlayers.size() - 1;

	if(use_gpu)
		return m_gpu_train.conv()[index].szW;
	return m_conv[index].szW;
}

int cifar_train::Kernels(int index, bool use_gpu)
{
	if(index >= (int)m_cnvlayers.size())
		index = (int)m_cnvlayers.size() - 1;

	if(use_gpu)
		return m_gpu_train.conv()[index].kernels;
	return m_conv[index].kernels;
}

int cifar_train::channels(int index, bool use_gpu)
{
	if(index >= (int)m_cnvlayers.size())
		index = (int)m_cnvlayers.size() - 1;

	if(use_gpu)
		return m_gpu_train.conv()[index].channels;
	return m_conv[index].channels;
}

void cifar_train::init_gpu()
{
	if(m_gpu_train.isInit())
		return;

	m_gpu_train.setConvLayers(m_cnvlayers, m_szA0);
	m_gpu_train.setMlpLayers(m_layers);

	m_gpu_train.init();
}

void cifar_train::setDebug(bool val)
{
	m_gpu_train.setDebug(val);
}

void cifar_train::setLambdaMlp(double val)
{
	m_gpu_train.setLambda(val);


	for(size_t i = 0; i < m_mlp.size(); ++i){
		ct::mlp<float>& m = m_mlp[i];
		m.setLambda(val);
	}
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

	read_vector(fs, m_cnvlayers);
	read_vector(fs, m_layers);

	fs.read((char*)&m_szA0, sizeof(m_szA0));

	setConvLayers(m_cnvlayers, m_szA0);

	init();

	for(size_t i = 0; i < m_conv.size(); ++i){
		conv2::convnn<float> &cnv = m_conv[i];
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

	write_vector(fs, m_cnvlayers);
	write_vector(fs, m_layers);

	fs.write((char*)&m_szA0, sizeof(m_szA0));

	for(size_t i = 0; i < m_conv.size(); ++i){
		conv2::convnn<float> &cnv = m_conv[i];
		cnv.write(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].write(fs);
	}

}

double cifar_train::statistics(int i0, int i1) const
{
	double v1 = m_statistics[i0 * count_classes + i1][0];
	double v2 = m_statistics[i0 * count_classes + i0][1];
	if(v2)
		return v1 / v2;
	return 0;
}

void cifar_train::setDropoutProb(double val)
{
	m_dropoutProb = val;
	m_gpu_train.setDropoutProb(val);
}

double cifar_train::dropoutProb() const
{
	return m_dropoutProb;
}

void cifar_train::save_weights(bool gpu)
{
	if(gpu){
		m_gpu_train.save_weights();
	}
}

void cifar_train::setDropout()
{
	for(int i = 0; i < (int)m_mlp.size() - 1; ++i){
		ct::mlpf& mlp = m_mlp[i];
		mlp.setDropout(true);
	}
}

void cifar_train::clearDropout()
{
	for(size_t i = 0; i < m_mlp.size(); ++i){
		ct::mlpf& mlp = m_mlp[i];
		mlp.setDropout(false);
	}
}
