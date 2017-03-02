#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "gpumat.h"
#include "cuda_common.h"
#include "common_types.h"

#include "common_devices.h"
#include "cuda_types.h"

using namespace gpumat;

///////// begin internal namespace ///////////////

namespace gpumat{

namespace internal{

template< typename T >
__device__ void _im2cols(const Mtx& X, const ct::Size& szA0, int channels, const ct::Size& szW, int stride, Mtx& Res, const ct::Size& szOut)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szOut.width * szOut.height;
	int all = szOutArea * channels;

	if(col < all){
		int c = col / szOutArea;
		int offset = col - c * szOutArea;

		int y = offset / szOut.width;
		int x = offset - y * szOut.width;

		int x0 = x * stride;
		int y0 = y * stride;
		int row2 = y * szOut.width + x;

		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dX = (T*)X.data;
		T *dR = (T*)Res.data;
		for(int c = 0; c < channels; ++c){
			T *dXi = &dX[c * szA0area];

			for(int a = 0; a < szW.height; ++a){
				for(int b = 0; b < szW.width; ++b){
					int col2 = c * szWarea + (a * szW.width + b);
					dR[row2 * Res.cols + col2] = dXi[(y0 + a) * szA0.width + (x0 + b)];
				}
			}
		}
	}
}

template< typename T >
__global__ void im2cols(Mtx X, ct::Size szA0, int channels, ct::Size szW, int stride, Mtx Res, ct::Size szOut)
{
	_im2cols<T>(X, szA0, channels, szW, stride, Res, szOut);
}

template< typename T >
__global__ void im2cols_vec(SmallMtxArray X, ct::Size szA0, int channels, ct::Size szW, int stride, SmallMtxArray Res, ct::Size szOut)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_im2cols<T>(X.mtx[row], szA0, channels, szW, stride, Res.mtx[row], szOut);
	}
}

////////

template< typename T >
__device__ void _back_deriv(const Mtx& Delta,
				 const ct::Size& szOut,
				 const ct::Size& szA0,
				 int channels,
				 const ct::Size& szW,
				 int stride,
				 Mtx X)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szOut.width * szOut.height;
	int all = szOutArea * channels;

	if(col < all){

		int c = col / szOutArea;
		int offset = col - c * szOutArea;

		int y = offset / szOut.width;
		int x = offset - y * szOut.width;

		int x0 = x * stride;
		int y0 = y * stride;
		int row2 = x * szOut.width + col;

		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dX = (T*)X.data;
		T *dR = (T*)Delta.data;
//		for(int c = 0; c < channels; ++c){
		T *dXi = &dX[c * szA0area];

		for(int a = 0; a < szW.height; ++a){
			for(int b = 0; b < szW.width; ++b){
				int col2 = c * szWarea + (a * szW.width + b);
				dXi[(y0 + a) * szA0.width + (x0 + b)] += dR[row2 * Delta.cols + col2];
			}
		}
//		}
	}
}

template< typename T >
__global__ void back_deriv(Mtx Delta,
						   ct::Size szOut,
						   ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   Mtx X)
{
	_back_deriv<T>(Delta, szOut, szA0, channels, szW, stride, X);
}

template< typename T >
__global__ void back_deriv_vec(SmallMtxArray Delta,
						   ct::Size szOut,
						   ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   SmallMtxArray X)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < Delta.count){
		_back_deriv<T>(Delta.mtx[row], szOut, szA0, channels, szW, stride, X.mtx[row]);
	}
}

template< typename T >
__device__ void _subsample(const Mtx &X,
						   int K,
						   const ct::Size& szA,
						   Mtx Y,
						   Mtx Mask,
						   const ct::Size& szO)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szO.width * szO.height;
	int all = szOutArea * K;

	const int stride = 2;

	if(col < all){
		int k = col / szOutArea;
		int offset = col - k * szOutArea;

		int y = offset / szO.width;
		int x = offset - y * szO.width;

		T *dX = (T*)X.data + k;
		T* dM = (T*)Mask.data + k;
		T *dY = (T*)Y.data + k;

		int y0 = y * stride;
		int x0 = x * stride;

		T mmax = dX[(y0 * szA.width + x0) * X.cols];
		int xm = x0, ym = y0;

		for(int a = 0; a < stride; ++a){
			for(int b = 0; b < stride; ++b){
				if(y0 + a < szA.height && x0 + b < szA.width){
					T val = dX[((y0 + a) * szA.width + (x0 + b)) * X.cols];
					if(val > mmax){
						mmax = val;
						xm = x0 + b;
						ym = y0 + a;
					}
				}
			}
		}

		dY[(y * szO.width + x) * Y.cols] = mmax;
		dM[(ym * szA.width + xm) * Mask.cols] = (T)1.;
	}
}

template< typename T >
__global__ void subsample(Mtx X,
						  int K,
						  ct::Size szA,
						  Mtx Y,
						  Mtx Mask,
						  ct::Size szO)
{
	_subsample<T>(X, K, szA, Y, Mask, szO);
}

template< typename T >
__global__ void subsample_vec(SmallMtxArray X,
						  int K,
						  ct::Size szA,
						  SmallMtxArray Y,
						  SmallMtxArray Mask,
						  ct::Size szO)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_subsample<T>(X.mtx[row], K, szA, Y.mtx[row], Mask.mtx[row], szO);
	}
}

template< typename T >
__device__ void _upsample(const Mtx &Y,
						 const Mtx &Mask,
						 int K,
						 const ct::Size &szO,
						 const ct::Size &szA,
						 Mtx X)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szO.width * szO.height;
	int all = szOutArea * K;

	const int stride = 2;

	if(col < all){
		int k = col / szOutArea;
		int offset = col - k * szOutArea;

		int y = offset / szO.width;
		int x = offset - y * szO.width;

		T *dX = (T*)X.data + k;
		T* dM = (T*)Mask.data + k;
		T *dY = (T*)Y.data + k;

		int y0 = y * stride;
		int x0 = x * stride;

		T val = dY[(y * szO.width + x) * Y.cols];

		for(int a = 0; a < stride; ++a){
			for(int b = 0; b < stride; ++b){
				if(y0 + a < szA.height && x0 + b < szA.width){
					T m = dM[((y0 + a) * szA.width + (x0 + b)) * Mask.cols];
					dX[((y0 + a) * szA.width + (x0 + b)) * X.cols] = val * m;
				}
			}
		}
	}
}

template< typename T >
__global__ void upsample(Mtx Y,
						 Mtx Mask,
						 int K,
						 ct::Size szO,
						 ct::Size szA,
						 Mtx X)
{
	_upsample<T>(Y, Mask, K, szO, szA, X);
}

template< typename T >
__global__ void upsample_vec(SmallMtxArray Y,
							 SmallMtxArray Mask,
							 int K,
							 ct::Size szO,
							 ct::Size szA,
							 SmallMtxArray X)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_upsample<T>(Y.mtx[row], Mask.mtx[row], K, szO, szA, X.mtx[row]);
	}
}

template< typename T >
__global__ void vec2mat(SmallMtxArray vec, Mtx mat)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < mat.rows && col < mat.cols){
		T* dV = (T*)vec.mtx[row].data;
		T* dM = (T*)mat.data;

		dM[row * mat.cols + col] = dV[col];
	}
}

template< typename T >
__global__ void mat2vec(Mtx mat, SmallMtxArray vec)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < mat.rows && col < mat.cols){
		T* dV = (T*)vec.mtx[row].data;
		T* dM = (T*)mat.data;

		dV[col] = dM[row * mat.cols + col];
	}
}

}	/// @endnamespace internal

}	/// @endnamespace gpumat

extern "C"
void cuda_im2cols(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::im2cols<double> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
		case GPU_FLOAT:
			internal::im2cols<float> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
	}
}

extern "C"
void cuda_im2cols_vec(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray sX(X), sRes(Res);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::im2cols_vec<double> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
		case GPU_FLOAT:
			internal::im2cols_vec<float> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
	}
}


extern "C"
void cuda_back_deriv(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::back_deriv<double> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::back_deriv<float> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
	}
}

extern "C"
void cuda_back_deriv_vec(const std::vector< gpumat::GpuMat > &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				std::vector< gpumat::GpuMat > &X)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = Delta.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray sDelta(Delta), sX(X);

	switch (Delta[0].type) {
		case GPU_DOUBLE:
			internal::back_deriv_vec<double> <<<dimGrid, dimBlock>>>(sDelta, szOut, szA0, channels, szW, stride, sX);
			break;
		case GPU_FLOAT:
			internal::back_deriv_vec<float> <<<dimGrid, dimBlock>>>(sDelta, szOut, szA0, channels, szW, stride, sX);
			break;
	}
}

extern "C"
void cuda_subsample2(const gpumat::GpuMat &X,
							  const ct::Size &szA,
							  gpumat::GpuMat &Y,
							  gpumat::GpuMat &Mask,
							  ct::Size &szO)
{
	int K = X.cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::subsample<double> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
		case GPU_FLOAT:
			internal::subsample<float> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
	}
}

extern "C"
void cuda_subsample2_vec(const std::vector< gpumat::GpuMat > &X,
					const ct::Size &szA,
					std::vector< gpumat::GpuMat > &Y,
					std::vector< gpumat::GpuMat > &Mask,
					ct::Size &szO)
{
	int K = X[0].cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::subsample_vec<double> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
		case GPU_FLOAT:
			internal::subsample_vec<float> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
	}
}

extern "C"
void cuda_upsample2(const gpumat::GpuMat &Y, const gpumat::GpuMat &Mask, const ct::Size &szO,
			  const ct::Size &szA, gpumat::GpuMat &X)
{
	int K = X.cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::upsample<double> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
		case GPU_FLOAT:
			internal::upsample<float> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
	}
}

extern "C"
void cuda_upsample2vec(const std::vector<gpumat::GpuMat> &Y, const std::vector<gpumat::GpuMat> &Mask,
			  const ct::Size &szO, const ct::Size &szA, std::vector<gpumat::GpuMat> &X)
{
	int K = X[0].cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::upsample_vec<double> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
		case GPU_FLOAT:
			internal::upsample_vec<float> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
	}
}


extern "C"
void cuda_vec2mat(const std::vector< GpuMat >& vec, GpuMat& mat)
{
	int rows = (int)vec.size();
	int cols = vec[0].total();

	int x1 = cols / BLOCKSIZE + 1;
	int x2 = rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (vec[0].type) {
		case GPU_DOUBLE:
			internal::vec2mat<double> <<<dimGrid, dimBlock>>>(vec, mat);
			break;
		case GPU_FLOAT:
			internal::vec2mat<float> <<<dimGrid, dimBlock>>>(vec, mat);
			break;
	}
}

extern "C"
void cuda_mat2vec(const GpuMat& mat, std::vector< GpuMat >& vec)
{
	int rows = mat.rows;
	int cols = mat.cols;

	int x1 = cols / BLOCKSIZE + 1;
	int x2 = rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (vec[0].type) {
		case GPU_DOUBLE:
			internal::mat2vec<double> <<<dimGrid, dimBlock>>>(mat, vec);
			break;
		case GPU_FLOAT:
			internal::mat2vec<float> <<<dimGrid, dimBlock>>>(mat, vec);
			break;
	}
}