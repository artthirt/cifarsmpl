#include "convnn2_gpu.h"
#include "nn.h"

using namespace gpumat::conv2;

convnn_gpu::convnn_gpu()
{

}

void convnn_gpu::setAlpha(double val)
{
	m_optim.setAlpha(val);
}

std::vector<gpumat::GpuMat> &convnn_gpu::XOut1()
{
	return A1;
}

std::vector<gpumat::GpuMat> &convnn_gpu::XOut2()
{
	return A2;
}

int convnn_gpu::outputFeatures() const
{
	if(m_use_pool){
		int val = szA2.area() * K;
		return val;
	}else{
		int val= szA1.area() * K;
		return val;
	}
}

ct::Size convnn_gpu::szOut() const
{
	if(m_use_pool)
		return szA2;
	else
		return szA1;
}

void convnn_gpu::init(const ct::Size &_szA0, int _channels, int stride, int _K, ct::Size &_szW, bool use_pool)
{
	szW = _szW;
	K = _K;
	channels = _channels;
	m_use_pool = use_pool;
	szA0 = _szA0;

	int rows = szW.area() * channels;
	int cols = K;

	ct::get_cnv_sizes(szA0, szW, stride, szA1, szA2);

	W.resize(1);
	B.resize(1);

	float n = (float)1./szW.area();
	for(size_t i = 0; i < W.size(); ++i){
		ct::Matf Wi(rows, cols), Bi(1, K);
		Wi.randn(0, n);
		gpumat::convert_to_gpu(Wi, W[i]);
		Bi.randn(0, n);
		gpumat::convert_to_gpu(Bi, B[i]);
	}

	m_optim.init(W, B);
}

void convnn_gpu::forward(const std::vector<gpumat::GpuMat> *_pX, gpumat::etypefunction func)
{
	if(!_pX)
		return;
	pX = (std::vector< gpumat::GpuMat >*)_pX;
	m_func = func;

	Xc.resize(pX->size());
	A1.resize(pX->size());

}

void convnn_gpu::backcnv(const std::vector<gpumat::GpuMat> &D, std::vector<gpumat::GpuMat> &DS)
{

}

void convnn_gpu::backward(const std::vector<gpumat::GpuMat> &D, bool last_level)
{
	if(D.empty() || D.size() != Xc.size()){
		throw new std::invalid_argument("vector D not complies saved parameters");
	}

}

///////////////////////////////
///////////////////////////////
///////////////////////////////

extern "C"
void cuda_im2cols(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut);

extern "C"
void cuda_im2cols_vec(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut);

extern "C"
void cuda_back_deriv(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X);

extern "C"
void cuda_back_deriv_vec(const std::vector< gpumat::GpuMat > &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				std::vector< gpumat::GpuMat > &X);

///////////////////////////////

void gpumat::conv2::im2cols(const gpumat::GpuMat &X, const ct::Size &szA0, int channels, const ct::Size &szW,
			 int stride, gpumat::GpuMat &Res, ct::Size &szOut)
{
	if(X.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");

	szOut.width = (szA0.width - szW.width)/stride + 1;
	szOut.height = (szA0.height - szW.height)/stride + 1;

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.resize(rows, cols, X.type);

	cuda_im2cols(X, szA0, channels, szW, stride, Res, szOut);
}


void gpumat::conv2::im2cols(const std::vector<gpumat::GpuMat> &X, const ct::Size &szA0, int channels, const ct::Size &szW, int stride, std::vector<gpumat::GpuMat> &Res, ct::Size &szOut)
{
	if(X.empty() || X[0].empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");
	szOut.width = (szA0.width - szW.width)/stride + 1;

	szOut.height = (szA0.height - szW.height)/stride + 1;

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.resize(X.size());

	for(size_t i = 0; i < Res.size(); ++i){
		Res[i].resize(rows, cols, X[i].type);
	}

	cuda_im2cols_vec(X, szA0, channels, szW, stride, Res, szOut);
}

void gpumat::conv2::back_deriv(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");

	X.resize(1, channels * szA0.area(), Delta.type);
	X.zeros();

	cuda_back_deriv(Delta, szOut, szA0, channels, szW, stride, X);
}


void gpumat::conv2::back_deriv(const std::vector<gpumat::GpuMat> &Delta, const ct::Size &szOut, const ct::Size &szA0, int channels, const ct::Size &szW, int stride, std::vector<gpumat::GpuMat> &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");

	X.resize(Delta.size());

	for(size_t i = 0; i < X.size(); ++i){
		X[i].resize(1, channels * szA0.area(), Delta[i].type);
		X[i].zeros();
	}

	cuda_back_deriv_vec(Delta, szOut, szA0, channels, szW, stride, X);

}
