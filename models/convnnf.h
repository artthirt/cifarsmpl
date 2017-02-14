#ifndef CONVNNF_H
#define CONVNNF_H

//#undef CONVNN_H

#include "custom_types.h"
#include "convnn.h"

typedef ct::convnn<float> convnnf;
typedef std::vector< convnnf > tvconvnnf;

//namespace convnn{

//	typedef convnn<float> convnnf;
//	typedef std::vector< convnnf > tvconvnnf;

//}

class ConvNN{
public:
	std::vector< int > m_cnvlayers;
	std::vector< int > m_cnvweights;
	std::vector< tvconvnnf > m_conv;
	ct::Size m_szA0;

	std::vector< ct::Matf > m_features;

	ConvNN();

	void setAlpha(float alpha);

	int outputFeatures() const;
	int outputMatrices() const;

	void init();
	void setConvLayers(const std::vector< int >& layers,
					   std::vector< int > weight_sizes,
					   const ct::Size szA0 = ct::Size(32, 32));
	void conv(const ct::Matf& X, ct::Matf& XOut);
	void backward(const ct::Matf& X);

	std::vector<tvconvnnf> &operator () ();
};


#endif // CONVNN_H
