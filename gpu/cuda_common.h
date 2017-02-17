#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include "gpumat.h"

#include <cuda_runtime.h>

/**
  size of block for cuda gpu
*/
#define BLOCKSIZE	32

namespace gpumat{

	enum etypefunction{
		RELU = 1,
		SIGMOID,
		SOFTMAX,
		TANH
	};

	namespace internal{

		struct Mtx{
			int rows;
			int cols;
			u_char* data;

			__host__ __device__ Mtx(){
				rows = cols = 0;
				data = 0;
			}
			__host__ __device__ Mtx(int rows, int cols, void* data){
				this->rows = rows;
				this->cols = cols;
				this->data = (u_char*)data;
			}
			__host__ __device__ Mtx(const gpumat::GpuMat& mat){
				rows = mat.rows;
				cols = mat.cols;
				data = mat.data;
			}

			__host__ __device__ int total() const{
				return rows * cols;
			}
		};

		struct SmallMtxArray{
//			enum {maxcount = 64};
			SmallMtxArray(){
				count = 0;
				mtx = 0;
			}
			~SmallMtxArray(){
				if(mtx){
					cudaFree(mtx);
				}
			}

			SmallMtxArray(const std::vector< GpuMat >& gmat){
//				if(maxcount < gmat.size())
//					throw new std::invalid_argument("not enough size of array for store matrices");

				count = gmat.size();

				size_t sz = sizeof(Mtx) * count;

				cudaMalloc(&mtx, sz);

				std::vector< Mtx > tmp;
				tmp.resize(count);

				for(size_t i = 0; i < count; ++i){
					tmp[i] = gmat[i];
				}
				cudaMemcpy(mtx, &tmp[0], sz, cudaMemcpyHostToDevice);
			}

			size_t count;
			internal::Mtx *mtx;
		};

		template< typename T>
		struct SmallSingleArray{
			enum {maxcount = 64};

			SmallSingleArray(){
				count = 0;
			}
			SmallSingleArray(const std::vector< T >& gv){
				if(maxcount < gv.size())
					throw new std::invalid_argument("not enough size of array for store matrices");

				count = gv.size();
				for(int i = 0; i < count; ++i){
					values[i] = gv[i];
				}
			}
			template< typename C >
			SmallSingleArray(const std::vector< C >& gv){
				if(maxcount < gv.size())
					throw new std::invalid_argument("not enough size of array for store matrices");

				count = gv.size();
				for(int i = 0; i < count; ++i){
					values[i] = gv[i];
				}
			}

			size_t count;
			T values[maxcount];
		};

		struct SmallMtxArrayStatic{
			enum {maxcount = 32};
			SmallMtxArrayStatic(){
				count = 0;
			}
			~SmallMtxArrayStatic(){
			}

			SmallMtxArrayStatic(const std::vector< GpuMat >& gmat, int beg, int last){
				if(maxcount < last - beg || last > gmat.size() || beg > gmat.size())
					throw new std::invalid_argument("not enough size of array for store matrices");

				count = last - beg;
				for(int i = beg, j = 0; i < last; ++i, ++j){
					mtx[j] = gmat[i];
				}
			}

			int count;
			internal::Mtx mtx[maxcount];
		};

	}/* @end internal */

}/* @end gpumat */

#endif // CUDA_COMMON_H
