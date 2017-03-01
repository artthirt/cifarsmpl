#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "gpumat.h"
#include "cuda_common.h"
#include "common_types.h"

#include "common_devices.h"

using namespace gpumat;

///////// begin internal namespace ///////////////

namespace gpumat{

namespace internal{

template< typename T >
__global__ void im2cols(Mtx X, ct::Size szA0, int channels, ct::Size szW, int stride, Mtx Res, ct::Size szOut)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(col < szOut.width && row < szOut.height){
		int x0 = col * stride;
		int y0 = row * stride;
		int row2 = row * szOut.width + col;

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

}

}

extern "C"
void cuda_im2cols(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.width / BLOCKSIZE + 1;
	int x2 = szOut.height / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::im2cols<double> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
		case GPU_FLOAT:
			internal::im2cols<float> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
	}

}
