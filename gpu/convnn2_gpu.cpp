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

	ct::Size szOut;

	gpumat::conv2::im2cols(*pX, szA0, channels, szW, stride, Xc, szOut);

	for(size_t i = 0; i < Xc.size(); ++i){
		gpumat::GpuMat& Xi = Xc[i];
		gpumat::GpuMat& A1i = A1[i];
		gpumat::matmul(Xi, W[0], A1i);
		gpumat::biasPlus(A1i, B[0]);
	}

	for(size_t i = 0; i < A1.size(); ++i){
		gpumat::GpuMat& Ao = A1[i];
		switch (m_func) {
			case gpumat::RELU:
				gpumat::reLu(Ao);
				break;
			case gpumat::SIGMOID:
				gpumat::sigmoid(Ao);
				break;
			case gpumat::TANH:
				gpumat::tanh(Ao);
				break;
			default:
				break;
		}
	}
	if(m_use_pool){
		Mask.resize(Xc.size());
		A2.resize(A1.size());
		ct::Size szOut;
		conv2::subsample(A1, szA1, A2, Mask, szOut);
		szK = A2[0].sz();
	}else{
		szK = A1[0].sz();
	}
}

void convnn_gpu::backcnv(const std::vector<gpumat::GpuMat> &D, std::vector<gpumat::GpuMat> &DS)
{
	DA1.resize(A1.size());
	if(D.data() != DS.data()){
		for(size_t i = 0; i < D.size(); ++i){
			switch (m_func) {
				case ct::RELU:
					gpumat::deriv_reLu(A1[i], DA1[i]);
					break;
				case ct::SIGMOID:
					gpumat::deriv_sigmoid(A1[i], DA1[i]);
					break;
				case ct::TANH:
					gpumat::deriv_tanh(A1[i], DA1[i]);
					break;
				default:
					break;
			}
			gpumat::elemwiseMult(D[i], DA1[i], DS[i]);
		}
	}else{
		for(size_t i = 0; i < D.size(); ++i){
			switch (m_func) {
				case ct::RELU:
					gpumat::deriv_reLu(A1[i], DA1[i]);
					break;
				case ct::SIGMOID:
					gpumat::deriv_sigmoid(A1[i], DA1[i]);
					break;
				case ct::TANH:
					gpumat::deriv_tanh(A1[i], DA1[i]);
					break;
				default:
					break;
			}
			gpumat::elemwiseMult(DS[i], DA1[i]);
		}
	}
}

void convnn_gpu::backward(const std::vector<gpumat::GpuMat> &D, bool last_level)
{
	if(D.empty() || D.size() != Xc.size()){
		throw new std::invalid_argument("vector D not complies saved parameters");
	}

	dSub.resize(D.size());

	if(m_use_pool){
		gpumat::conv2::upsample(D, Mask, szA2, szA1, dSub);
		backcnv(dSub, dSub);
	}else{
		backcnv(D, dSub);
	}

	vgW.resize(D.size());
	vgB.resize(D.size());
	for(size_t i = 0; i < D.size(); ++i){
		gpumat::GpuMat& Xci = Xc[i];
		gpumat::GpuMat& dSubi = dSub[i];
		gpumat::GpuMat& Wi = vgW[i];
		gpumat::GpuMat& vgBi = vgB[i];
		gpumat::matmulT1(Xci, dSubi, Wi);

//		gB.swap_dims();
		sumRows(dSubi, vgBi, 1.f / dSubi.rows);
//		gB.swap_dims();

//		vgBi = (ct::sumRows(dSubi)) * (1.f/dSubi.rows);
		//Wi *= (1.f/dSubi.total());
		//vgBi.swap_dims();
	}
	gW.resize(1);
	gB.resize(1);
	gW[0].resize(W[0]);
	gW[0].zeros();
	gB[0].resize(B[0]);
	gB[0].zeros();
	for(size_t i = 0; i < D.size(); ++i){
		gpumat::add(gW[0], vgW[i]);
		gpumat::add(gB[0], vgB[i]);
	}
	gpumat::mulval(gW[0], 1./(D.size()));
	gpumat::mulval(gB[0], 1./(D.size()));

	if(!last_level){
		Dlt.resize(D.size());

		//ct::Mat_<T> Wf;
		//flipW(W, szW, channels, Wf);

		Dc.reserve(D.size());
		for(size_t i = 0; i < D.size(); ++i){
			gpumat::GpuMat& Dci = Dc[i];
			gpumat::matmulT2(dSub[i], W[0], Dci);
			//ct::Size sz = (*pX)[i].size();
			//Dlt[i].set_dims(sz);
		}
		back_deriv(Dc, szA1, szA0, channels, szW, stride, Dlt);
	}

	m_optim.pass(gW, gB, W, B);
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

extern "C"
void cuda_subsample2(const gpumat::GpuMat &X,
					const ct::Size &szA,
					gpumat::GpuMat &Y,
					gpumat::GpuMat &Mask,
					ct::Size &szO);

extern "C"
void cuda_subsample2_vec(const std::vector< gpumat::GpuMat > &X,
					const ct::Size &szA,
					std::vector< gpumat::GpuMat > &Y,
					std::vector< gpumat::GpuMat > &Mask,
					ct::Size &szO);

extern "C"
void cuda_vec2mat(const std::vector< gpumat::GpuMat >& vec, gpumat::GpuMat& mat);

extern "C"
void cuda_mat2vec(const gpumat::GpuMat& mat, std::vector< gpumat::GpuMat >& vec);

extern "C"
void cuda_upsample2(const gpumat::GpuMat &Y, const gpumat::GpuMat &Mask, const ct::Size &szO,
			  const ct::Size &szA, gpumat::GpuMat &X);

extern "C"
void cuda_upsample2vec(const std::vector<gpumat::GpuMat> &Y, const std::vector<gpumat::GpuMat> &Mask,
			  const ct::Size &szO, const ct::Size &szA, std::vector<gpumat::GpuMat> &X);

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

void gpumat::conv2::subsample(const gpumat::GpuMat &X,
							  const ct::Size &szA,
							  gpumat::GpuMat &Y,
							  gpumat::GpuMat &Mask,
							  ct::Size &szO)
{
	if(X.empty() || X.rows != szA.area())
		throw new std::invalid_argument("subsample: empty parameters");

	szO.width = szA.width / 2;
	szO.height = szA.height / 2;
	int K = X.cols;

	Y.resize(szO.area(), K, X.type);
	Mask.resize(X.rows, X.cols, X.type);
	Mask.zeros();

	cuda_subsample2(X, szA, Y, Mask, szO);
}

void gpumat::conv2::subsample(const std::vector<gpumat::GpuMat> &X,
							  const ct::Size &szA,
							  std::vector<gpumat::GpuMat> &Y,
							  std::vector<gpumat::GpuMat> &Mask,
							  ct::Size &szO)
{
	if(X.empty() || X[0].rows != szA.area())
		throw new std::invalid_argument("subsample: empty parameters");

	szO.width = szA.width / 2;
	szO.height = szA.height / 2;
	int K = X[0].cols;

	Y.resize(X.size());
	Mask.resize(X.size());

	for(size_t i = 0; i < X.size(); ++i){
		Y[i].resize(szO.area(), K, X[i].type);
		Mask[i].resize(X[i].rows, X[i].cols, X[i].type);
		Mask[i].zeros();
	}

	cuda_subsample2_vec(X, szA, Y, Mask, szO);
}

void gpumat::conv2::vec2mat(const std::vector<gpumat::GpuMat> &vec, gpumat::GpuMat &mat)
{
	if(vec.empty() || vec[0].empty())
		throw new std::invalid_argument("vec2mat: empty parameters");

	int rows = (int)vec.size();
	int cols = vec[0].total();

	mat.resize(rows, cols, vec[0].type);

	cuda_vec2mat(vec, mat);
}

void gpumat::conv2::mat2vec(const gpumat::GpuMat &mat, const ct::Size &szOut, std::vector<gpumat::GpuMat> &vec)
{
	if(mat.empty())
		throw new std::invalid_argument("mat2vec: empty parameters");

	int rows = mat.rows;

	vec.resize(rows);

	for(size_t i = 0; i < vec.size(); ++i){
		vec[i].resize(szOut.height, szOut.width, mat.type);
	}

	cuda_mat2vec(mat, vec);
}

void gpumat::conv2::upsample(const gpumat::GpuMat &Y, const gpumat::GpuMat &Mask, const ct::Size &szO,
			  const ct::Size &szA, gpumat::GpuMat &X)
{
	if(Y.empty() || Mask.empty() || Y.rows != szO.area())
		throw new std::invalid_argument("mat2vec: empty parameters");

	int K = Y.cols;
	X.resize(szA.area(), K, Y.type);

	cuda_upsample2(Y, Mask, szO, szA, X);
}

void gpumat::conv2::upsample(const std::vector<gpumat::GpuMat> &Y, const std::vector<gpumat::GpuMat> &Mask,
			  const ct::Size &szO, const ct::Size &szA, std::vector<gpumat::GpuMat> &X)
{
	if(Y.empty() || Y[0].empty() || Mask.empty() || Mask[0].empty() || Y[0].rows != szO.area())
		throw new std::invalid_argument("mat2vec: empty parameters");

	int K = Y[0].cols;

	X.resize(Y.size());

	for(size_t i = 0; i < X.size(); ++i){
		X[i].resize(szA.area(), K, Y[0].type);
	}

	cuda_upsample2vec(Y, Mask, szO, szA, X);

}
