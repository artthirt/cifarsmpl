#ifndef GPU_MLP_H
#define GPU_MLP_H

#include "custom_types.h"
#include "gpumat.h"
#include "cuda_common.h"
#include "helper_gpu.h"

namespace gpumat{

class mlp{
public:
	GpuMat *pA0;
	GpuMat W;
	GpuMat B;
	GpuMat Z;
	GpuMat A1;
//	GpuMat DA1;
	GpuMat PartZ;
	GpuMat DltA0;
	GpuMat Dropout;
	GpuMat WDropout;
	GpuMat gW;
	GpuMat gB;

	mlp();

	void setDropout(bool val, double p = 0.95);

	bool isInit() const;

	void init(int input, int output, int type);

	inline void apply_func(const GpuMat& Z, GpuMat& A, etypefunction func);
	inline void apply_func(GpuMat& A, etypefunction func);
	inline void apply_back_func(const GpuMat& D1, GpuMat& D2, etypefunction func);

	etypefunction funcType() const;

	void forward(const GpuMat *mat, etypefunction func = RELU, bool save_A0 = true);

	void backward(const GpuMat &Delta, bool last_layer = false);

	void write(std::fstream& fs);
	void read(std::fstream& fs);

private:
	bool m_init;
	bool m_is_dropout;
	double m_prob;
	etypefunction m_func;
};

class MlpOptim: public AdamOptimizer{
public:
	void init(const std::vector< gpumat::mlp >& _mlp);
	bool pass(std::vector<mlp> &_mlp);
private:
};

}

#endif // GPU_MLP_H
