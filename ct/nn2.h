#ifndef NN2_H
#define NN2_H

#include "custom_types.h"
#include "matops.h"

namespace ct{

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
						int col = c * (a *szW.width + b);
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dR[row * Res.cols + col] = dXi[(y0 + a) * szA0.width + (x0 + b)];
						}
					}
				}

			}
		}
	}
}

}

#endif // NN2_H
