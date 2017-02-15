#ifndef CIFAR_TRAIN_H
#define CIFAR_TRAIN_H

#include "custom_types.h"
#include "cifar_reader.h"
#include "convnn.h"
#include "mlp.h"
#include "convnnf.h"

#include "gpu_train.h"

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
					   const ct::Size szA0 = ct::Size(32, 32));
	void setMlpLayers(const std::vector< int >& layers);

	void init();

	void forward(const std::vector< ct::Matf >& X, ct::Matf& a_out,
				 bool use_drop = false, float p = 0.92, bool use_gpu = false);

	void pass(int batch, bool use_gpu = false, std::vector<double> *percents = nullptr);

	void getEstimage(int batch, double& accuracy, double& l2, bool use_gpu = false);

	void setAlpha(double alpha);

	uint iteration() const;

	uint iteration_gpu() const;

	QVector<int> predict(const QVector<TData> &data, bool use_gpu = false);

	QVector<QVector<ct::Matf> > cnvW(int index, bool use_gpu = false);

	void init_gpu();

private:
	cifar_reader* m_cifar;
	std::vector< int > m_layers;
	std::vector< int > m_cnvlayers;
	std::vector< int > m_cnvweights;
	ct::Size m_szA0;
	bool m_init;

	gpu_train m_gpu_train;

	ct::Matf m_tX;
	ct::Matf m_ty;
	ct::Matf m_td;
	ct::Matf m_X_out;

	std::vector< ct::Matf > m_splitD;

	std::vector< ConvNN > m_conv;
	std::vector< ct::mlpf > m_mlp;
	ct::MlpOptim< float > m_optim;

	void setDropout(float p, int layers);
	void clearDropout();
};

#endif // CIFAR_TRAIN_H
