#ifndef CIFAR_TRAIN_H
#define CIFAR_TRAIN_H

#include "custom_types.h"
#include "cifar_reader.h"
#include "mlp.h"

#include "convnn2.h"

#include "gpu_train.h"

#include <QMap>

namespace ct{
	typedef mlp<float> mlpf;
}

class cifar_train
{
public:
	cifar_train();

	void setCifar(cifar_reader* val);

	void setConvLayers(const std::vector< ct::ParamsCnv >& layers,
					   const ct::Size szA0 = ct::Size(32, 32));
	void setMlpLayers(const std::vector< ct::ParamsMlp >& layers);

	void init();

	void forward(const std::vector< ct::Matf >& X, ct::Matf& a_out,
				 bool use_drop = false, bool use_gpu = false);

	void setRandData(float offset, float angle, float br);

	void setUseRandData(bool val);

	void pass(int batch, bool use_gpu = false);

	void getEstimate(int batch, double& accuracy, double& l2, bool use_gpu = false);
	void getEstimateTest(int batch, double& accuracy, double& l2, bool use_gpu = false);

	void getEstimateTest(double& accuracy, double& l2, bool use_gpu = false);

	void setAlpha(double alpha);
	void setAlphaCnv(double val);

	uint iteration() const;

	uint iteration_gpu() const;

	uint inputToMlp(bool use_gpu = false) const;

	QVector<int> predict(const QVector<TData> &data, bool use_gpu = false);

	ct::Matf cnvW(int index, bool use_gpu = false);
	ct::Size& szW(int index, bool use_gpu = false);
	int Kernels(int index, bool use_gpu = false);
	int channels(int index, bool use_gpu = false);

	void init_gpu();

	void setDebug(bool val);

	void setLambdaMlp(double val);

	bool loadFromFile(const QString& fn, bool gpu);
	void saveToFile(const QString& fn, bool gpu);

	double statistics(int i0, int i1) const;

	void setDropoutProb(double val);
	double dropoutProb() const;

	void save_weights(bool gpu);

private:
	cifar_reader* m_cifar;
	std::vector< ct::ParamsMlp > m_layers;
	std::vector< ct::ParamsCnv > m_cnvlayers;
	ct::Size m_szA0;
	bool m_init;
	double m_dropoutProb;

	std::vector< ct::Vec2i> m_statistics;

	gpu_train m_gpu_train;

	ct::Matf m_tX;
	ct::Matf m_ty;
	ct::Matf m_td;
	ct::Matf m_X_out;

	std::vector< ct::Matf > m_splitD;

	std::vector< conv2::convnn<float> > m_conv;
	std::vector< ct::mlpf > m_mlp;
	ct::MlpOptimMoment< float > m_optim;
	conv2::CnvMomentumOptimizer< float > m_cnv_optim;

	std::vector< ct::Vec4f > m_vals;

	bool m_use_rand_data;
	ct::Vec3f m_rand_data;

	void setDropout();
	void clearDropout();
	void randValues(size_t count, std::vector< ct::Vec4f >& vals, float offset, float angle, float brightness = 0.1);
	void randX(std::vector<ct::Matf> &X, std::vector<ct::Vec4f> &vals);

	void getEstimate(const std::vector< ct::Matf > &Xs, ct::Matf &y,
					 uint &right, double &l2, bool use_gpu);

	void sliceEstimage(const std::vector< ct::Matf > &Xs, ct::Matf &y,
					   uint &right, double &l2);

	int getRight(const ct::Matf& y, const ct::Matf& yp);

};

#endif // CIFAR_TRAIN_H
