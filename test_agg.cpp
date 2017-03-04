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
#include "gpu_mlp.h"

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

		cnv2.forward(&cnv1.A2, ct::RELU);

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

		cnv2.backward(D);

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

class TestCnv_gpu{
public:
	TestCnv_gpu(){
		mlp.resize(3);
		int K1 = 15;
		int K2 = 35;

		szA0 = ct::Size(cifar_reader::WidthIM, cifar_reader::HeightIM);
		szW = ct::Size(5, 5);

		cnv1.init(szA0, 3, 1, K1, szW);
		cnv2.init(cnv1.szA2, K1, 1, K2, szW);

	}

	gpumat::conv2::convnn_gpu cnv1, cnv2;
	gpumat::MlpOptim optim;
	std::vector< gpumat::mlp > mlp;

	ct::Size szA0, szW;

	std::vector< gpumat::GpuMat > Xout, Xout2;

	void forward(const std::vector< gpumat::GpuMat > &Xs, gpumat::GpuMat **yp, bool dropout = true)
	{
		cnv1.forward(&Xs, gpumat::RELU);

		cnv2.forward(&cnv1.A2, gpumat::RELU);

		gpumat::conv2::vec2mat(cnv2.A2, X1);

		if(!mlp[0].isInit()){
			mlp[0].init(X1.cols, 700, gpumat::GPU_FLOAT);
			mlp[1].init(700, 600, gpumat::GPU_FLOAT);
			mlp[2].init(600, 10, gpumat::GPU_FLOAT);
			optim.init(mlp);
		}

		mlp[0].setDropout(dropout, 0.98f);
		mlp[1].setDropout(dropout, 0.98f);

		mlp[0].forward(&X1, gpumat::RELU);
		mlp[1].forward(&mlp[0].A1, gpumat::RELU);
		mlp[2].forward(&mlp[1].A1, gpumat::SOFTMAX);

		*yp = &mlp[2].A1;
	}

	void backward(const gpumat::GpuMat& dy)
	{
		mlp[2].backward(dy);
		mlp[1].backward(mlp[2].DltA0);
		mlp[0].backward(mlp[1].DltA0);

		optim.pass(mlp);
		gpumat::conv2::mat2vec(mlp[0].DltA0, cnv2.szK, D);

		cnv2.backward(D);

		cnv1.backward(cnv2.Dlt, true);
	}

	void test(const std::vector< gpumat::GpuMat > &Xs, const ct::Matf& yind, double &l2, double &acc){
		ct::Matf redf, ypf;

		gpumat::GpuMat *yp;
		forward(Xs, &yp, false);

		gpumat::convert_to_gpu(yind, g_yind);

		gpumat::subIndOne(*yp, g_yind, dy);

		gpumat::elemwiseSqr(dy, dy2);
		gpumat::reduce(dy2, red);
		gpumat::convert_to_mat(red, redf);
		l2 = redf.at(0, 0) / dy2.rows;

		gpumat::convert_to_mat(*yp, ypf);

		int count = 0;
		for(int i = 0; i < ypf.rows; ++i){
			int idp = ypf.argmax(i, 1);
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
	std::vector< gpumat::GpuMat > D;
	gpumat::GpuMat X1, g_yind;

	gpumat::GpuMat dy, dy2, red;
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

	conv2::upsample(Y, channels, Mask, szOut2, szOut, Z2);
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

	gpumat::GpuMat g_Z, g_W, g_Y, g_Mask;
	std::vector< gpumat::GpuMat > g_vZ, g_vY, g_vMask, g_vX2;

	gpumat::convert_to_gpu(W, g_W);

//	gpumat::matmul(g_Res, g_W, g_Z);
	g_vZ.resize(vRes.size());
	g_vY.resize(vRes.size());
	g_vMask.resize(vRes.size());
	for(size_t i = 0; i < vRes.size(); ++i){
		gpumat::matmul(vRes[i], g_W, g_vZ[i]);
	}
	gpumat::conv2::subsample(g_vZ, szOut, g_vY, g_vMask, szOut2);
	ct::Size szK = g_vY[0].sz();

	ct::Matf Y2, Mask2;

	gpumat::convert_to_mat(g_vY[0], Y2);
	gpumat::convert_to_mat(g_vMask[0], Mask2);

	ct::save_mat(Y2, "Y2.txt");
	ct::save_mat(Mask2, "Mask2.txt");

	gpumat::conv2::upsample(g_vY, K, g_vMask, szOut2, szOut, g_vZ);
	gpumat::convert_to_mat(g_vZ[0], Y2);
	ct::save_mat(Y2, "gZ2.txt");

	for(int i = 0; i < 10; i++){
		gpumat::conv2::upsample(g_vY, K, g_vMask, szOut2, szOut, g_vZ);
		gpumat::conv2::back_deriv(g_vZ, szOut, szA0, channels, szW, 1, g_vX2);
	}
	gpumat::convert_to_mat(g_vX2[0], Y2);
	ct::save_mat(Y2, "g_Deriv.txt");

	gpumat::conv2::vec2mat(g_vY, g_Y);
	gpumat::convert_to_mat(g_Y, Y2);
	ct::save_mat(Y2, "vec2mat.txt");

	gpumat::conv2::mat2vec(g_Y, szK, g_vY);
	gpumat::convert_to_mat(g_vY[0], Y2);
	ct::save_mat(Y2, "mat2vec.txt");

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

void conv_vec_to_gpu(const std::vector< ct::Matf >& Xs, std::vector< gpumat::GpuMat >& g_Xs)
{
	g_Xs.resize(Xs.size());
	for(int i = 0; i < Xs.size(); ++i){
		gpumat::convert_to_gpu(Xs[i], g_Xs[i]);
	}
}

void test_agg::test_conv_gpu()
{
	cifar_reader rd;
	rd.openDir("../../../data/cifar-10-batches-bin");
	//rd.openDir("D:/Down/smpl/data/cifar-10-batches-bin");
	if(!rd.isBinDataExists())
		return;

	std::vector< ct::Matf > Xs;
	std::vector< gpumat::GpuMat > g_Xs, g_XsTest;
	ct::Matf y, yp;
	gpumat::GpuMat g_y, g_yTest, g_tmp, g_red;

	gpumat::GpuMat g_dy;

	ShowMatrices sh;

	ct::generator.seed(1);

	TestCnv_gpu tcnv;
//	rd.getTrain2(70, Xs, y);

	for(int i = 0; i < 15000; ++i){

		rd.getTrain2(70, Xs, y);

		gpumat::convert_to_gpu(y, g_y);
		conv_vec_to_gpu(Xs, g_Xs);

		gpumat::GpuMat *pyp;

		tcnv.forward(g_Xs, &pyp);

		gpumat::subIndOne(*pyp, g_y, g_dy);

		tcnv.backward(g_dy);

		if((i % 10) == 0){
			ct::Matf dy;
			gpumat::elemwiseSqr(g_dy, g_tmp);
			gpumat::reduce(g_tmp, g_red);
			gpumat::convert_to_mat(g_red, dy);
			double sy2 = (double)dy.sum()/g_tmp.rows;
			qDebug("pass %d: l2 = %f", i, sy2);
		}
//		continue;

		if((i % 30) == 0){
			{
				ct::Matf W1, W2;
				gpumat::convert_to_mat(tcnv.cnv1.W[0], W1);
				gpumat::convert_to_mat(tcnv.cnv2.W[0], W2);
				sh.saveMat("cnv1.bmp", W1, tcnv.cnv1.szW, tcnv.cnv1.K, tcnv.cnv1.channels);
				sh.saveMat("cnv2.bmp", W2, tcnv.cnv2.szW, tcnv.cnv2.K, tcnv.cnv2.channels);
			}

			rd.getTrain2(500, Xs, y);
			double l2, acc;

			conv_vec_to_gpu(Xs, g_XsTest);
			//gpumat::convert_to_gpu(y, g_yTest);

			tcnv.test(g_XsTest, y, l2, acc);
			qDebug("train (batch=500) -> l2=%f,\tacc=%f", l2, acc);

			l2 = 0; acc = 0;
			y.fill(0);
			rd.getTest2(500, Xs, y);

			conv_vec_to_gpu(Xs, g_XsTest);
			gpumat::convert_to_gpu(y, g_yTest);

			tcnv.test(g_XsTest, y, l2, acc);
			qDebug("test  (batch=500) -> l2=%f,\tacc=%f", l2, acc);
		}
	}

}

template< typename T >
void check_zero(const ct::Mat_<T>& mat)
{
	ct::Mat_<T> tmp2;
	tmp2 = ct::elemwiseSqr(mat);
	assert(tmp2.sum() < 1e-6);
}

void test_agg::test_file()
{
	ct::Matf A0, Xc, W, B, A1, A2, tmp1, tmp2, tmp3, D, dSub1, dSub2, Mask;
	ct::Size szOut;

	qt_work_mat::q_load_mat("testPx26.txt", A0);
	qt_work_mat::q_load_mat("testXc26.txt", Xc);
	qt_work_mat::q_load_mat("testA126.txt", A1);
	qt_work_mat::q_load_mat("testA226.txt", A2);
	qt_work_mat::q_load_mat("testW.txt", W);
	qt_work_mat::q_load_mat("testB.txt", B);
	qt_work_mat::q_load_mat("testMask.txt", Mask);

	qt_work_mat::q_load_mat("testD26.txt", D);
	qt_work_mat::q_load_mat("testDSub26.txt", dSub1);
	qt_work_mat::q_load_mat("testDSub2_26.txt", dSub2);

	conv2::im2col(A0, ct::Size(32, 32), 3, ct::Size(5, 5), 1, tmp1, szOut);
	check_zero(tmp1 - Xc);

	tmp2 = Xc * W;
	tmp2.biasPlus(B);
	ct::v_relu(tmp2);
	check_zero(tmp2 - A1);

	conv2::subsample(tmp2, ct::Size(28, 28), tmp1, tmp3, szOut);
	check_zero(tmp1 - A2);
	check_zero(tmp3 - Mask);

	conv2::upsample(D, 15, Mask, ct::Size(14, 14), ct::Size(28, 28), tmp1);
	check_zero(tmp1 - dSub1);
	ct::elemwiseMult(tmp1, ct::derivRelu(A1));
	check_zero(tmp1 - dSub2);
}
