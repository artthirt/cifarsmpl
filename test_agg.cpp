#include "test_agg.h"

#include <iostream>

#include "custom_types.h"
#include "gpumat.h"
#include "convnn_gpu.h"

#include "convnn2.h"
#include "qt_work_mat.h"

#include <fstream>

template< typename T >
void saveimage2file(const ct::Mat_<T>& X, const ct::Size &szA, int channels, const std::string& fn)
{
	std::fstream fs;
	fs.open(fn, std::ios_base::out);

	fs << std::setprecision(4);
	for(int c = 0; c < channels; ++c){
		T *dXi = X.ptr() + c * szA.area();
		for(int y = 0; y < szA.height; ++y){
			for(int x = 0; x < szA.width; ++x){
				fs << dXi[y * szA.width + x] << "\t";
			}
			fs << std::endl;
		}
		fs << "------" << std::endl;
	}
	fs.close();
}

test_agg::test_agg()
{
	ct::Size szA0(14, 14), szW(5, 5), szOut, szOut2;
	int channels = 3;

	ct::Matf X(1, szA0.area() * channels), Res, Z, Z2, Y, W, Mask;
	for(int i = 0; i < X.total(); ++i){
		X.ptr()[i] = i/10.;
	}

	int K = 5;

	W.setSize(szW.area() * channels, K);
	W.fill(1.);

	saveimage2file(X, szA0, channels, "X.txt");

	conv2::im2col(X, szA0, channels, szW, 1, Res, szOut);

	ct::save_mat(Res, "Res.txt");

	Z = Res * W;
	ct::save_mat(W, "W.txt");
	ct::save_mat(Z, "Z.txt");

	conv2::subsample(Z, szOut, Y, Mask, szOut2);

	ct::save_mat(Y, "Y.txt");
	ct::save_mat(Mask, "Mask.txt");

	conv2::upsample(Y, Mask, szOut2, szOut, Z2);
	ct::save_mat(Z2, "Z2.txt");
}

void test_agg::test_hconcat()
{
	ct::Matf mat(160, 400);
	mat.randn(0, 1);

//	std::string smat = mat;

	std::cout << "Original matrix\n" << mat.print() << std::endl;

	gpumat::GpuMat gmat, gout;
	std::vector< gpumat::GpuMat > glist;

	gpumat::convert_to_gpu(mat, gmat);

	gpumat::hsplit(gmat, 40, glist);

	gout.zeros();

	gpumat::hconcat(glist, gout);

	std::cout << "Output matrix\n" << gout.print() << std::endl;
}

void test_agg::test_im2col()
{

}
