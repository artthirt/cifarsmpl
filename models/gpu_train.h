#ifndef GPU_TRAIN_H
#define GPU_TRAIN_H

#include "custom_types.h"
#include "gpu_mlp.h"
#include "convnn_gpu.h"
#include "convnn2_gpu.h"

class gpu_train
{
public:
	gpu_train();

	void setConvLayers(const std::vector< int >& layers,
					   std::vector< int > weight_sizes,
					   const ct::Size szA0 = ct::Size(32, 32),
					   std::vector<char> *pooling = nullptr);
	void setMlpLayers(const std::vector< int >& layers);

	void setAlpha(double alpha);
	void setAlphaCnv(double alpha);

	void init();
	bool isInit() const;

	uint iteration() const;

	double getL2(const ct::Matf &yp, const ct::Matf &y);

	void forward(const std::vector< ct::Matf > &X, ct::Matf &a_out,
				 bool use_drop = false, double p = 0.95, bool use_ret = true);

	void forward(const std::vector< gpumat::GpuMat > &X, gpumat::GpuMat **pAout,
				 bool use_drop = false, double p = 0.95);
	void pass(const std::vector< ct::Matf >& X, const ct::Matf &y);
	void pass();

	bool loadFromFile(const std::string& fn);
	void saveToFile(const std::string& fn);

	uint outputFeatures() const;

private:
	std::vector< int > m_layers;
	std::vector< int > m_cnvlayers;
	std::vector< char > m_cnvpooling;
	std::vector< int > m_cnvweights;
	ct::Size m_szA0;
	bool m_init;

	std::vector< gpumat::conv2::convnn_gpu > m_conv;
	std::vector< gpumat::mlp > m_mlp;
	gpumat::MlpOptim m_optim;

	std::vector< gpumat::GpuMat > m_XsIn;
	std::vector< gpumat::GpuMat > m_Xs;
	std::vector< gpumat::GpuMat > m_splitD;
	gpumat::GpuMat m_y_ind;
	gpumat::GpuMat m_y_ind2;
	gpumat::GpuMat m_gyp;
	gpumat::GpuMat m_red;
	gpumat::GpuMat m_yp;
	gpumat::GpuMat m_td;
	gpumat::GpuMat m_tsub;
	gpumat::GpuMat m_Xout;

	void setDropout(float p, int layers);
	void clearDropout();
};

#endif // GPU_TRAIN_H
