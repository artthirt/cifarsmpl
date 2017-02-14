#ifndef CIFAR_TRAIN_H
#define CIFAR_TRAIN_H

#include "custom_types.h"
#include "cifar_reader.h"
#include "convnn.h"
#include "mlp.h"
#include "convnnf.h"

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
				 bool use_drop = false, float p = 0.92);

	void pass(int batch);

	void getEstimage(int batch, double& accuracy, double& l2);

	void setAlpha(double alpha);

	uint iteration() const;

	QVector<int> predict(double percent, int batch);

	QVector<QVector<ct::Matf> > cnvW(int index);

private:
	cifar_reader* m_cifar;
	std::vector< int > m_layers;
	std::vector< int > m_cnvlayers;
	std::vector< int > m_cnvweights;
	ct::Size m_szA0;
	bool m_init;

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
