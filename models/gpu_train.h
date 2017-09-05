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

	void setConvLayers(const std::vector< ct::ParamsCnv >& layers,
					   const ct::Size szA0 = ct::Size(32, 32));
	void setMlpLayers(const std::vector< ct::ParamsMlp >& layers);

	void setDebug(bool val);
	void setLambda(double val);

	void setAlpha(double alpha);
	void setAlphaCnv(double alpha);

	void init();
	bool isInit() const;

	uint iteration() const;

	double getL2(const ct::Matf &yp, const ct::Matf &y);

	void forward(const std::vector< ct::Matf > &X, ct::Matf &a_out,
				 bool use_drop = false, bool use_ret = true);

	void forward(const std::vector< gpumat::GpuMat > &X, gpumat::GpuMat **pAout,
				 bool use_drop = false);
	void pass(const std::vector< ct::Matf >& X, const ct::Matf &y);
	void pass();

	bool loadFromFile(const std::string& fn);
	void saveToFile(const std::string& fn);

	uint outputFeatures() const;

	std::vector<gpumat::convnn_gpu> &conv();

	void setDropoutProb(double val);
	double dropoutProb() const;

	void save_weights();

private:
	std::vector< ct::ParamsMlp > m_layers;
	std::vector< ct::ParamsCnv > m_cnvlayers;
	ct::Size m_szA0;
	bool m_init;
	double m_dropoutProb;

	std::vector< gpumat::convnn_gpu > m_conv;
	std::vector< gpumat::mlp > m_mlp;
	gpumat::MlpOptimMoment m_optim;
	gpumat::CnvMomentumOptimizer m_cnv_optim;

	std::vector< gpumat::GpuMat > m_XsIn;
	std::vector< gpumat::GpuMat > m_Xs;
	std::vector< gpumat::GpuMat > m_splitD;
	gpumat::GpuMat m_y_ind;
	gpumat::GpuMat m_y_ind2;
	gpumat::GpuMat m_gyp;
	gpumat::GpuMat m_yp;
	gpumat::GpuMat m_td;
	gpumat::GpuMat m_tsub;
	gpumat::GpuMat m_Xout;

	bool m_is_debug;

	void setDropout();
	void clearDropout();
};

#endif // GPU_TRAIN_H
