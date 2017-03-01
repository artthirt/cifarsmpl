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

	void setConvLayers(const std::vector< int >& layers,
					   std::vector< int > weight_sizes,
					   const ct::Size szA0 = ct::Size(32, 32),
					   std::vector<char> *pooling = nullptr);
	void setMlpLayers(const std::vector< int >& layers);

	void init();

	void forward(const std::vector< ct::Matf >& X, ct::Matf& a_out,
				 bool use_drop = false, float p = 0.92, bool use_gpu = false);

	void setRandData(float offset, float angle);

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

	ct::Matf &cnvW(int index, bool use_gpu = false);
	ct::Size& szW(int index);
	int Kernels(int index);
	int channels(int index);

	void init_gpu();

	bool loadFromFile(const QString& fn, bool gpu);
	void saveToFile(const QString& fn, bool gpu);

	ct::Vec2i statistics(int val) const;

private:
	cifar_reader* m_cifar;
	std::vector< int > m_layers;
	std::vector< int > m_cnvlayers;
	std::vector< int > m_cnvweights;
	std::vector< char > m_cnvpooling;
	ct::Size m_szA0;
	bool m_init;

	QMap< int, ct::Vec2i > m_statistics;

	gpu_train m_gpu_train;

	ct::Matf m_tX;
	ct::Matf m_ty;
	ct::Matf m_td;
	ct::Matf m_X_out;

	std::vector< ct::Matf > m_splitD;

	std::vector< conv2::convnn<float> > m_conv;
	std::vector< ct::mlpf > m_mlp;
	ct::MlpOptim< float > m_optim;

	std::vector< ct::Vec3f > m_vals;

	bool m_use_rand_data;
	ct::Vec2f m_rand_data;

	void setDropout(float p, int layers);
	void clearDropout();
	void randValues(size_t count, std::vector< ct::Vec3f >& vals, float offset, float angle);
	void randX(ct::Matf &X, std::vector< ct::Vec3f >& vals);

	void getEstimate(const std::vector< ct::Matf > &Xs, ct::Matf &y,
					 uint &right, double &l2, bool use_gpu);

	void sliceEstimage(const std::vector< ct::Matf > &Xs, ct::Matf &y,
					   uint &right, double &l2);

	int getRight(const ct::Matf& y, const ct::Matf& yp);

};

#endif // CIFAR_TRAIN_H
