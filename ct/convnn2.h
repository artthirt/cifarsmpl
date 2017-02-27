#ifndef NN2_H
#define NN2_H

#include "custom_types.h"
#include "matops.h"

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

	szOut.width = (szA0.width - szW.width)/stride + 1;
	szOut.height = (szA0.height - szW.height)/stride + 1;

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

}

#endif // NN2_H
