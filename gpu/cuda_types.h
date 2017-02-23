#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

#include <vector>

namespace gpumat{

class GpuMat;

namespace internal{

class Mtx;

struct SmallMtxArray{
//			enum {maxcount = 64};
	SmallMtxArray();
	~SmallMtxArray();

	SmallMtxArray(const std::vector< GpuMat >& gmat);

	void set(const std::vector< GpuMat >& gmat);

	void copyFrom(const SmallMtxArray& mt);

	void setDelete(bool val);

	size_t count;
	size_t allocate;
	size_t size;
	internal::Mtx *mtx;
	bool m_delete;
};

}

}

#endif // CUDA_TYPES_H
