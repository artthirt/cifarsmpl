#include "convnnf.h"
#include "nn.h"

using namespace convnn;

ConvNN::ConvNN()
{

}

void ConvNN::setAlpha(float alpha)
{
	for(size_t i = 0; i < m_conv.size(); ++i){
		for(size_t j = 0; j < m_conv[i].size(); ++j){
			convnnf& cnv = m_conv[i][j];
			cnv.setAlpha(alpha);
		}
	}
}

int ConvNN::outputFeatures() const
{
	return m_conv.back()[0].W.size() * m_conv.back()[0].szOut().area() * m_conv.back().size();
}

int ConvNN::outputMatrices() const
{
	return m_conv.back()[0].W.size() * m_conv.back().size();
}

void ConvNN::init()
{
	if(m_cnvlayers.empty() || m_cnvweights.empty())
		throw new std::invalid_argument("empty arguments");

	m_conv.resize(m_cnvlayers.size());

	ct::Size szA0 =m_szA0;

	int input = 1;
	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].resize(input);

		bool pool = true;
		if(m_cnvpooling.size())
			pool = m_cnvpooling[i];

		for(size_t j = 0; j < m_conv[i].size(); ++j){
			convnnf& cnv = m_conv[i][j];
			cnv.setWeightSize(m_cnvweights[i], pool);
			cnv.init(m_cnvlayers[i], szA0, pool);
		}
		input = m_cnvlayers[i] * input;
		szA0 = m_conv[i][0].szOut();
	}
}

void ConvNN::setConvLayers(const std::vector<int> &layers,
						   std::vector<int> weight_sizes,
						   const ct::Size szA0,
						   std::vector<char> *pooling)
{
	if(layers.empty() || weight_sizes.empty())
		throw new std::invalid_argument("empty parameters");

	if(pooling)
		m_cnvpooling = *pooling;
	m_cnvlayers = layers;
	m_cnvweights = weight_sizes;
	m_szA0 = szA0;
}

void ConvNN::conv(const ct::Matf &X, ct::Matf &XOut)
{
//	ct::Matf *pX = (ct::Matf*)&X;

	for(size_t i = 0; i < m_conv.size(); ++i){
		std::vector< convnnf >& ls = m_conv[i];

		for(size_t j = 0; j < m_conv[i].size(); ++j){
			//convnnf& cnv = m_conv[i][j];

			if(i == 0){
				convnnf& m0 = ls[0];
				m0.forward(&X, ct::RELU);
			}else{
//				int jm1 = j / m_cnvlayers[i];
//				int k = j - jm1 * m_cnvlayers[i];
//				convnnf& pcnv = m_conv[i - 1][jm1];
//				cnv.forward(&pcnv.A2[k], ct::RELU);
#pragma omp parallel for
				for(int j = 0; j < m_conv[i - 1].size(); ++j){
					size_t off1 = j * m_cnvlayers[i - 1];
					convnnf& m0 = m_conv[i - 1][j];
					for(int k = 0; k < m_cnvlayers[i - 1]; ++k){
						size_t col = off1 + k;
						convnnf& mi = ls[col];
						if(m0.use_pool())
							mi.forward(&m0.A2[k], ct::RELU);
						else
							mi.forward(&m0.A1[k], ct::RELU);
					}
				}
			}
		}
	}
	convnnf::hconcat(m_conv.back(), XOut);
}

void ConvNN::backward(const ct::Matf &X)
{
	if(m_cnvlayers.empty() || m_cnvweights.empty())
		throw new std::invalid_argument("empty arguments");

	int cols = m_conv.back().size() * m_conv.back()[0].W.size();

	ct::hsplit(X, cols, m_features);

	for(int i = m_conv.size() - 1; i > -1; i--){
		std::vector< convnnf >& lrs = m_conv[i];

//			qDebug("LR[%d]-----", i);
		size_t kidx = 0;

		for(size_t j = 0; j < lrs.size(); ++j){
			convnnf &cnv = lrs[j];

			size_t kfirst = kidx;
			kidx += cnv.W.size();

			if(i == m_conv.size() - 1)
				cnv.backward(m_features, kfirst, kidx, i == 0);
			else
				cnv.backward(m_conv[i + 1], kfirst, kidx, i == 0);
		}
//			qDebug("----");
	}
}

void ConvNN::write(std::fstream& fs)
{
	if(!fs.is_open() || !m_conv.size())
		return;

	for(size_t i = 0; i < m_conv.size(); ++i){
		for(size_t j = 0; j < m_conv[i].size(); ++j){
			convnnf& cnv = m_conv[i][j];
			cnv.write(fs);
		}
	}
}

void ConvNN::read(std::fstream& fs)
{
	if(!fs.is_open() || !m_conv.size())
		return;

	for(size_t i = 0; i < m_conv.size(); ++i){
		for(size_t j = 0; j < m_conv[i].size(); ++j){
			convnnf& cnv = m_conv[i][j];
			cnv.read(fs);
		}
	}
}

std::vector<tvconvnnf> &ConvNN::operator ()()
{
	return m_conv;
}
