#include "test_agg.h"

#include <iostream>
#include <assert.h>

#include "custom_types.h"
#include "gpumat.h"
#include "convnn_gpu.h"

#include "convnn2.h"
#include "qt_work_mat.h"

#include <fstream>

#include "cifar_reader.h"
#include "mlp.h"

#include "convnn2_gpu.h"

#include "showmatrices.h"

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

/////////////////////
/////////////////////

class TestCnv{
public:
	TestCnv(){
		mlp.resize(3);
		int K1 = 25;
		int K2 = 15;

		szA0 = ct::Size(cifar_reader::WidthIM, cifar_reader::HeightIM);
		szW = ct::Size(5, 5);

		cnv1.init(szA0, 3, 1, K1, szW);
		cnv2.init(cnv1.szA2, K1, 1, K2, szW);

	}

	conv2::convnn<float> cnv1, cnv2;
	ct::MlpOptim< float > optim;
	std::vector< ct::mlp<float> > mlp;

	ct::Size szA0, szW;

	void forward(const std::vector< ct::Matf > &Xs, ct::Matf& yp, bool dropout = true)
	{
		std::vector< ct::Matf > Xout, Xout2;

		cnv1.forward(&Xs, ct::RELU);

//		Xout.resize(cnv1.A2.size());
//		for(size_t i = 0; i < cnv1.A2.size(); ++i){
//			Xout[i] = cnv1.A2[i].t();
//			Xout[i].set_dims(1, cnv1.A2[i].total());
//		}

		cnv2.forward(&cnv1.A2, ct::RELU);

//		Xout2.resize(cnv2.A2.size());
//		for(size_t i = 0; i < cnv2.A2.size(); ++i){
//			Xout2[i] = cnv2.A2[i].t();
//			Xout2[i].set_dims(1, cnv2.A2[i].total());
//		}

		conv2::vec2mat(cnv2.A2, X1);

		if(!mlp[0].isInit()){
			mlp[0].init(X1.cols, 700);
			mlp[1].init(700, 600);
			mlp[2].init(600, 10);
			optim.init(mlp);
		}

		mlp[0].setDropout(dropout, 0.9f);
		mlp[1].setDropout(dropout, 0.9f);

		mlp[0].forward(&X1, ct::RELU);
		mlp[1].forward(&mlp[0].A1, ct::RELU);
		mlp[2].forward(&mlp[1].A1, ct::SOFTMAX);

		yp = mlp.back().A1;
	}

	void backward(const ct::Matf dy)
	{
		mlp[2].backward(dy);
		mlp[1].backward(mlp[2].DltA0);
		mlp[0].backward(mlp[1].DltA0);

		optim.pass(mlp);
		D.clear();
		conv2::mat2vec(mlp[0].DltA0, cnv2.szK.t(), D);

//		for(size_t i = 0; i < Xout2.size(); ++i){
//			D[i] = D[i].t();
//		}

		cnv2.backward(D);

//		for(size_t i = 0; i < Xout2.size(); ++i){
//			cnv2.Dlt[i].set_dims(cnv1.szK.t());
//			cnv2.Dlt[i] = cnv2.Dlt[i].t();
//		}

		cnv1.backward(cnv2.Dlt, true);
	}

	void test(const std::vector< ct::Matf > &Xs, const ct::Matf& yind, double &l2, double &acc){
		ct::Matf yp, dy;
		forward(Xs, yp, false);
		dy = ct::subIndOne(yp, yind);

		ct::Matf dy2 = ct::elemwiseSqr(dy);
		l2 = dy2.sum() / dy2.rows;

		int count = 0;
		for(int i = 0; i < yp.rows; ++i){
			int idp = yp.argmax(i, 1);
			int idy = yind.at(i, 0);
//			ct::Vec2i vec = m_statistics[idy];
			if(idy == idp){
				count++;
//				vec[0]++;
			}
//			vec[1]++;
//			m_statistics[idy] = vec;
		}
		acc = (double)1. * count / yind.rows;

	}

private:
	std::vector< ct::Matf > Xout, Xout2, D;
	ct::Matf X1;
};

/////////////////////
/////////////////////

test_agg::test_agg()
{

}

/**
 * @brief test_agg::test_hconcat
 */
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

/**
 * @brief test_agg::test_im2col
 */
void test_agg::test_im2col()
{
	ct::Size szA0(14, 14), szW(5, 5), szOut, szOut2;
	int channels = 3;

	ct::Matf X(1, szA0.area() * channels), Res, Z, Z2, Y, W, Mask;
	for(int i = 0; i < X.total(); ++i){
		X.ptr()[i] = i;
	}

	int K = 3;

	W.setSize(szW.area() * channels, K);
	W.fill(1.);
	for(int i = 0; i < W.total(); ++i){
		W.ptr()[i] = i;
	}

	saveimage2file(X, szA0, channels, "X.txt");

	conv2::im2col(X, szA0, channels, szW, 1, Res, szOut);

	ct::save_mat(Res, "Res.txt");

	Z = Res * W;
	ct::save_mat(W.t(), "W.txt");
	ct::save_mat(Z, "Z.txt");

	ct::Matf Wr;
	conv2::flipW(W, szW, channels, Wr);
	ct::save_mat(Wr.t(), "Wr.txt");

	conv2::subsample(Z, szOut, Y, Mask, szOut2);

	ct::save_mat(Y, "Y.txt");
	ct::save_mat(Mask, "Mask.txt");

	conv2::upsample(Y, Mask, szOut2, szOut, Z2);
	ct::save_mat(Z2, "Z2.txt");

	///////////////////

	gpumat::GpuMat g_X, g_Res;

	gpumat::convert_to_gpu(X, g_X);

	gpumat::conv2::im2cols(g_X, szA0, channels, szW, 1, g_Res, szOut);

	ct::Matf Res2, y;

	gpumat::convert_to_mat(g_Res, Res2);

	ct::save_mat(Res2, "Res_g.txt");

	y = Res - Res2;
	y = ct::elemwiseSqr(y);
	assert(y.sum() == 0);

	std::vector< gpumat::GpuMat > vX, vRes;
	vX.push_back(g_X);
	vX.push_back(g_X);

	gpumat::conv2::im2cols(vX, szA0, channels, szW, 1, vRes, szOut);

	for(size_t i = 0; i < vRes.size(); ++i){
		gpumat::convert_to_mat(vRes[i], Res2);

		y = Res - Res2;
		y = ct::elemwiseSqr(y);
		assert(y.sum() == 0);
		ct::save_mat(Res2, "vRes_g.txt");
	}
}

void test_agg::test_conv()
{
	cifar_reader rd;
	rd.openDir("../../../data/cifar-10-batches-bin");
	//rd.openDir("D:/Down/smpl/data/cifar-10-batches-bin");
	if(!rd.isBinDataExists())
		return;

	std::vector< ct::Matf > Xs;
	ct::Matf y, yp, X1, dy;

	ShowMatrices sh;

	ct::generator.seed(time(0));

	TestCnv tcnv;

	for(int i = 0; i < 15000; ++i){

		rd.getTrain2(70, Xs, y);

		tcnv.forward(Xs, yp);

		dy = ct::subIndOne(yp, y);

		if((i % 10) == 0){
			sh.saveMat("cnv1.bmp", tcnv.cnv1.W, tcnv.cnv1.szW, tcnv.cnv1.K, tcnv.cnv1.channels);
			sh.saveMat("cnv2.bmp", tcnv.cnv2.W, tcnv.cnv2.szW, tcnv.cnv2.K, tcnv.cnv2.channels);

			ct::Matf dy2 = ct::elemwiseSqr(dy);
			double sy2 = (double)dy2.sum()/dy2.rows;
			qDebug("pass %d: l2 = %f", i, sy2);
		}

		tcnv.backward(dy);

		if((i % 30) == 0){
			rd.getTrain2(500, Xs, y);
			double l2, acc;
			tcnv.test(Xs, y, l2, acc);
			qDebug("train (batch=500) -> l2=%f,\tacc=%f", l2, acc);

			l2 = 0; acc = 0;
			Xs.clear();
			y.fill(0);
			rd.getTest2(500, Xs, y);
			tcnv.test(Xs, y, l2, acc);
			qDebug("test  (batch=500) -> l2=%f,\tacc=%f", l2, acc);
		}
	}
}
