#include "test_agg.h"

#include <iostream>

#include "custom_types.h"
#include "gpumat.h"
#include "convnn_gpu.h"

#include "convnn2.h"
#include "qt_work_mat.h"

#include <fstream>

#include "cifar_reader.h"
#include "mlp.h"

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
	ct::Size szA0(14, 14), szW(5, 5), szOut, szOut2;
	int channels = 10;

	ct::Matf X(1, szA0.area() * channels), Res, Z, Z2, Y, W, Mask;
	for(int i = 0; i < X.total(); ++i){
		X.ptr()[i] = i/1000.;
	}

	int K = 3;

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

void test_agg::test_conv()
{
	cifar_reader rd;
	rd.openDir("D:/Down/smpl/data/cifar-10-batches-bin");
	if(!rd.isBinDataExists())
		return;

	conv2::convnn<float> cnv1, cnv2;

	std::vector< ct::Matf > Xs, Xout, Xout2, D;
	ct::Matf y, X1, dy;

	ct::mlp<float> mlp, mlp1;

	ct::Size szA0(cifar_reader::WidthIM, cifar_reader::HeightIM), szW(5, 5);

	int K1 = 5;
	int K2 = 10;

	cnv1.init(szA0, 3, 1, K1, szW);
	//cnv2.init(cnv1.szA2, K1, 1, K2, szW);

	rd.getTrain2(10, Xs, y);

	for(int i = 0; i < 1000; ++i){

		cnv1.forward(&Xs, ct::RELU, Xout);
//		cnv2.forward(&Xout, ct::RELU, Xout2);

		conv2::vec2mat(Xout, X1);

		if(!mlp.isInit()){
			mlp.init(X1.cols, 100);
			mlp1.init(100, 10);
		}

		mlp.forward(&X1, ct::RELU);
		mlp1.forward(&mlp.A1, ct::SOFTMAX);

		dy = ct::subIndOne(mlp1.A1, y);

		ct::Matf dy2 = ct::elemwiseSqr(dy);
		float sy2 = dy2.sum();
		qDebug("l2 = %f", sy2);

		mlp1.backward(dy);
		mlp.backward(mlp1.DltA0);

		D.clear();
		conv2::mat2vec(mlp.DltA0, cnv1.szK, D);

//		cnv2.backward(D);
		cnv1.backward(D, true);
	}
}
