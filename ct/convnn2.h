#ifndef NN2_H
#define NN2_H

#include "custom_types.h"
#include "matops.h"
#include <vector>
#include "nn.h"

namespace conv2{

template< typename T >
void im2col(const ct::Mat_<T>& X, const ct::Size& szA0, int channels, const ct::Size& szW, int stride, ct::Mat_<T>& Res, ct::Size& szOut)
{
	if(X.empty() || !channels)
		return;

	szOut.width = (szA0.width - szW.width)/stride + 1;
	szOut.height = (szA0.height - szW.height)/stride + 1;

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.setSize(rows, cols);

	T *dX = X.ptr();
	T *dR = Res.ptr();
	for(int c = 0; c < channels; ++c){
		T *dXi = &dX[c * szA0.area()];
		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

				for(int a = 0; a < szW.height; ++a){
					for(int b = 0; b < szW.width; ++b){
						int col = c * szW.area() + (a * szW.width + b);
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dR[row * Res.cols + col] = dXi[(y0 + a) * szA0.width + (x0 + b)];
						}
					}
				}

			}
		}
	}
}

template< typename T >
void back_deriv(const ct::Mat_<T>& Delta, const ct::Mat_<T>& W, const ct::Size& szOut, const ct::Size& szA0, int channels, const ct::Size& szW, int stride, ct::Mat_<T>& X)
{
	if(Delta.empty() || !channels)
		return;

	X.setSize(1, channels * szA0.area());
	X.fill(0);

	T *dX = X.ptr();
	T *dR = Delta.ptr();
	T *dW = W.ptr();
	for(int c = 0; c < channels; ++c){
		T *dXi = &dX[c * szA0.area()];
		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

				T *dWi = &dW[c * szW.area()];

				for(int a = 0; a < szW.height; ++a){
					for(int b = 0; b < szW.width; ++b){
						int col = c * szW.area() + (a * szW.width + b);
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dXi[(y0 + a) * szA0.width + (x0 + b)] += dR[row * Delta.cols + col] * dWi[a * szW.width + b];
						}
					}
				}

			}
		}
	}
}

template< typename T >
void subsample(const ct::Mat_<T>& X, const ct::Size& szA, ct::Mat_<T>& Y, ct::Mat_<T>& Mask, ct::Size& szO)
{
	if(X.empty() || X.rows != szA.area())
		return;

	szO.width = szA.width / 2;
	szO.height = szA.height / 2;
	int K = X.cols;

	Y.setSize(szO.area(), K);
	Mask.setSize(X.size());
	Mask.fill(0);

	int stride = 2;

	for(int k = 0; k < K; ++k){
		T *dX = X.ptr() + k;
		T* dM = Mask.ptr() + k;
		T *dY = Y.ptr() + k;

		for(int y = 0; y < szO.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szO.width; ++x){
				int x0 = x * stride;

				T mmax = dX[(y0 * szA.width + x0) * X.cols];
				int xm = x0, ym = y0;
				for(int a = 0; a < stride; ++a){
					for(int b = 0; b < stride; ++b){
						if(y0 + a < szA.height && x0 + b < szA.width){
							T val = dX[((y0 + a) * szA.width + (x0 + b)) * X.cols];
							if(val > mmax){
								mmax = val;
								xm = x0 + b;
								ym = y0 + a;
							}
						}
					}
				}

				dY[(y * szO.width + x) * Y.cols] = mmax;
				dM[(ym * szA.width + xm) * Mask.cols] = 1.;
			}
		}
	}
}

template< typename T >
void upsample(const ct::Mat_<T>& Y, const ct::Mat_<T>& Mask, const ct::Size& szO, const ct::Size& szA, ct::Mat_<T>& X)
{
	if(Y.empty() || Mask.empty() || Y.rows != szO.area())
		return;

	int K = Y.cols;
	X.setSize(szA.area(), K);

	int stride = 2;

	for(int k = 0; k < K; ++k){
		T *dX = X.ptr() + k;
		T* dM = Mask.ptr() + k;
		T *dY = Y.ptr() + k;

		for(int y = 0; y < szO.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szO.width; ++x){
				int x0 = x * stride;

				T val = dY[(y * szO.width + x) * Y.cols];

				for(int a = 0; a < stride; ++a){
					for(int b = 0; b < stride; ++b){
						if(y0 + a < szA.height && x0 + b < szA.width){
							T m = dM[((y0 + a) * szA.width + (x0 + b)) * Mask.cols];
							dX[((y0 + a) * szA.width + (x0 + b)) * X.cols] = val * m;
						}
					}
				}
			}
		}
	}
}

template< typename T >
void vec2mat(const std::vector< ct::Mat_<T> >& vec, ct::Mat_<T>& mat)
{
	if(vec.empty() || vec[0].empty())
		return;

	int rows = vec.size();
	int cols = vec[0].total();

	mat.setSize(rows, cols);

	T *dM = mat.ptr();
	for(size_t i = 0; i < rows; ++i){
		const ct::Mat_<T>& V = vec[i];
		T *dV = V.ptr();
		for(int j = 0; j < V.total(); ++j){
			dM[i * cols + j] = dV[j];
		}
	}
}

template< typename T >
void mat2vec(const ct::Mat_<T>& mat, const ct::Size& szOut, std::vector< ct::Mat_<T> >& vec)
{
	if(mat.empty())
		return;

	int rows = mat.rows;
	int cols = mat.cols;

	vec.resize(rows);

	T *dM = mat.ptr();
	for(size_t i = 0; i < rows; ++i){
		ct::Mat_<T>& V = vec[i];
		V.setSize(szOut);
		T *dV = V.ptr();
		for(int j = 0; j < V.total(); ++j){
			dV[j] = dM[i * cols + j];
		}
	}
}

//-------------------------------------

template< typename T >
class convnn{
public:
	ct::Mat_<T> W;
	ct::Mat_<T> B;
	int K;
	int channels;
	int stride;
	ct::Size szA0;
	ct::Size szA1;
	ct::Size szA2;
	ct::Size szW;
	ct::Size szK;
	std::vector< ct::Mat_<T> >* pX;
	std::vector< ct::Mat_<T> > Xc;
	std::vector< ct::Mat_<T> > Z;
	std::vector< ct::Mat_<T> > Dlt;
	std::vector< ct::Mat_<T> > vgW;
	std::vector< ct::Mat_<T> > vgB;
	std::vector< ct::Mat_<T> > Mask;
	ct::AdamOptimizer< T > m_optim;

	ct::Mat_<T> gW;
	ct::Mat_<T> gB;

	convnn(){
		m_use_pool = false;
		stride = 1;
	}

	void init(const ct::Size& _szA0, int _channels, int stride, int _K, ct::Size& _szW, bool use_pool = true){
		szW = _szW;
		K = _K;
		channels = _channels;
		m_use_pool = use_pool;
		szA0 = _szA0;

		int rows = szW.area() * channels;
		int cols = K;

		ct::get_cnv_sizes(szA0, szW, stride, szA1, szA2);

		T n = (T)1./szW.area();

		W.setSize(rows, cols);
		W.randn(0, n);
		B.setSize(1, K);
		B.randn(0, n);

		std::vector< ct::Mat_<T> > vW, vB;
		vW.push_back(W);
		vB.push_back(B);
		m_optim.init(vW, vB);
	}

	void forward(std::vector< ct::Mat_<T> >* _pX, ct::etypefunction func, std::vector< ct::Mat_<T> >& Xout){
		if(!_pX)
			return;
		pX = _pX;
		m_func = func;

		Xc.resize(pX->size());
		Z.resize(pX->size());
		Xout.resize(pX->size());

		for(size_t i = 0; i < Xc.size(); ++i){
			ct::Mat_<T>& Xi = (*pX)[i];
			ct::Size szOut;

			im2col(Xi, szA0, channels, szW, stride, Xc[i], szOut);
		}

		for(size_t i = 0; i < Xout.size(); ++i){
			ct::Mat_<T>& Xi = Xc[i];
			ct::Mat_<T>& Zi = Z[i];
			ct::Mat_<T>& Xo = Xout[i];
			Zi = Xi * W;
			Zi.biasPlus(B);

			switch (m_func) {
				case ct::RELU:
					ct::v_relu(Zi, Xo);
					break;
				case ct::SIGMOID:
					ct::v_sigmoid(Zi, Xo);
					break;
				case ct::TANH:
					ct::v_tanh(Zi, Xo);
					break;
				default:
					break;
			}

		}
		if(m_use_pool){
			Mask.resize(Xc.size());
			for(size_t i = 0; i < Xout.size(); ++i){
				ct::Mat_<T> &Xo = Xout[i], Y;
				ct::Size szOut;
				conv2::subsample(Xo, szA1, Y, Mask[i], szOut);
				Xo = Y;
			}
		}
		szK = Xout[0].size();
	}

	inline void backcnv(const std::vector< ct::Mat_<T> >& D, std::vector< ct::Mat_<T> >& DS){
		if(D.data() != DS.data()){
			for(size_t i = 0; i < D.size(); ++i){
				switch (m_func) {
					case ct::RELU:
						ct::v_derivRelu(D[i], DS[i]);
						break;
					case ct::SIGMOID:
						ct::v_derivSigmoid(D[i], DS[i]);
						break;
					case ct::TANH:
						ct::v_derivTanh(D[i], DS[i]);
						break;
					default:
						break;
				}
			}
		}else{
			for(size_t i = 0; i < D.size(); ++i){
				switch (m_func) {
					case ct::RELU:
						ct::v_derivRelu(DS[i]);
						break;
					case ct::SIGMOID:
						ct::v_derivSigmoid(DS[i]);
						break;
					case ct::TANH:
						ct::v_derivTanh(DS[i]);
						break;
					default:
						break;
				}
			}
		}
	}

	void backward(const std::vector< ct::Mat_<T> >& D, bool last_level = false){
		if(D.empty() || D.size() != Mask.size())
			return;

		std::vector< ct::Mat_<T> > dSub;
		dSub.resize(D.size());

		if(m_use_pool){
			for(size_t i = 0; i < D.size(); ++i){
				const ct::Mat_<T>& Di = D[i];
				upsample(Di, Mask[i], szA2, szA1, dSub[i]);
			}
			backcnv(dSub, dSub);
		}else{
			backcnv(D, dSub);
		}

		vgW.resize(D.size());
		vgB.resize(D.size());
		for(size_t i = 0; i < D.size(); ++i){
			ct::Mat_<T>& Xci = Xc[i];
			ct::Mat_<T>& dSubi = dSub[i];
			ct::Mat_<T>& Wi = vgW[i];
			ct::Mat_<T>& vgBi = vgB[i];
			matmulT1(Xci, dSubi, Wi);
			vgBi = (ct::sumRows(dSubi)) * (1.f/dSubi.rows);
			//Wi *= (1.f/dSubi.rows);
			//vgBi.swap_dims();
		}
		gW.setSize(W.size());
		gW.fill(0);
		gB.setSize(B.size());
		gB.fill(0);
		for(size_t i = 0; i < D.size(); ++i){
			gW += vgW[i];
			gB += vgB[i];
		}
//		gW *= 1./D.size();
//		gB *= 1./D.size();

		std::vector< ct::Mat_<T>> vgW, vgB, vW, vB;
		vgW.push_back(gW);
		vW.push_back(W);
		vgB.push_back(gB);
		vB.push_back(B);

		if(!last_level){
			Dlt.resize(D.size());
			for(size_t i = 0; i < D.size(); ++i){
				ct::Matf Dc;
				ct::matmulT2(W, dSub[i], Dc);
				back_deriv(Dc, W, szA1, szA0, channels, szW, stride, Dlt[i]);
				ct::Size sz = (*pX)[i].size();
				Dlt[i].set_dims(sz);
			}
		}

		m_optim.pass(vgW, vgB, vW, vB);
		W = vW[0]; B = vB[0];

	}

private:
	bool m_use_pool;
	ct::etypefunction m_func;
};

}

#endif // NN2_H
