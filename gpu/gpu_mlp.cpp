#include "gpu_mlp.h"

#include "matops.h"

using namespace gpumat;

template< typename T >
inline void init(GpuMat& mat, T n){
	ct::Mat_<T> m;
	m.setSize(mat.rows, mat.cols);
	m.randn(0, n);
	gpumat::convert_to_gpu(m, mat);
}

////////////////////////////////////////////////

mlp::mlp(){
	m_func = RELU;
	m_init = false;
	m_is_dropout = false;
	m_prob = 0.95;
	pA0 = nullptr;
}

void mlp::setDropout(bool val, double p){
	m_is_dropout = val;
	m_prob = p;
}

bool mlp::isInit() const
{
	return m_init;
}

void mlp::init(int input, int output, int type){
	double n = 1./sqrt(input);

	W.resize(input, output, type);
	B.resize(output, 1, type);

	switch (type) {
		case GPU_DOUBLE:
			::init<double>(W, n);
			::init<double>(B, n);
			break;
		case GPU_FLOAT:
			::init<float>(W, (float)n);
			::init<float>(B, (float)n);
			break;
	}

	m_init = true;
}

void mlp::apply_func(const GpuMat &Z, GpuMat &A, etypefunction func){
	switch (func) {
		default:
		case RELU:
			reLu(Z, A);
			break;
		case SOFTMAX:
			softmax(Z, 1, A, PartZ);
			break;
		case SIGMOID:
			sigmoid(Z, A);
			break;
		case TANH:
			tanh(Z, A);
			break;
	}
}

void mlp::apply_func(GpuMat &A, etypefunction func)
{
	switch (func) {
		default:
		case RELU:
			reLu(A);
			break;
		case SOFTMAX:
			softmax(A, 1, PartZ);
			break;
		case SIGMOID:
			sigmoid(A);
			break;
		case TANH:
			tanh(A);
			break;
	}
}

void mlp::apply_back_func(const GpuMat &D1, GpuMat &D2, etypefunction func){
	switch (func) {
		default:
		case RELU:
			deriv_reLu(A1, DA1);
			break;
		case SOFTMAX:
			//				A = softmax(A, 1);
			D1.copyTo(D2);
			return;
		case SIGMOID:
			deriv_sigmoid(A1, DA1);
			break;
		case TANH:
			deriv_tanh(A1, DA1);
			break;
	}
	elemwiseMult(D1, DA1, D2);
}

etypefunction mlp::funcType() const{
	return m_func;
}

inline void apply_dropout(const GpuMat& W, double prob, GpuMat& WDropout, GpuMat& Dropout)
{
	switch (W.type) {
		case GPU_DOUBLE:{
			ct::Matd _Dropout;
			ct::dropout(W.rows, W.cols, prob, _Dropout);
			convert_to_gpu(_Dropout, Dropout);
			elemwiseMult(W, Dropout, WDropout);
			break;
		}
		case GPU_FLOAT:{
			ct::Matf _Dropout;
			ct::dropout(W.rows, W.cols, (float)prob, _Dropout);
			convert_to_gpu(_Dropout, Dropout);
			elemwiseMult(W, Dropout, WDropout);
			break;
		}
	}
}

void mlp::forward(const GpuMat *mat, etypefunction func, bool save_A0)
{
	if(!m_init || !mat)
		throw new std::invalid_argument("mlp::forward: not initialized. wrong parameters");
	pA0 = (GpuMat*)mat;
	m_func = func;

	if(m_is_dropout){
		apply_dropout(W, m_prob, WDropout, Dropout);
		matmul(*pA0, WDropout, A1);
	}else{
		matmul(*pA0, W, A1);
	}

	biasPlus(A1, B);
	apply_func(A1, A1, func);

	if(!save_A0)
		pA0 = nullptr;
}

void mlp::backward(const GpuMat &Delta, bool last_layer)
{
	if(!pA0 || !m_init)
		throw new std::invalid_argument("mlp::backward: not initialized. wrong parameters");

	apply_back_func(Delta, DA1, m_func);

	double m = Delta.rows;

	matmulT1(*pA0, DA1, gW);
	mulval(gW, 1. / m);

	if(m_is_dropout){
		elemwiseMult(gW, Dropout);
	}

	gB.swap_dims();
	sumRows(DA1, gB, 1.f / m);
	gB.swap_dims();

	if(!last_layer){
		matmulT2_shared(DA1, W, DltA0);
	}
}

void mlp::write(std::fstream &fs)
{
	gpumat::write_fs(fs, W);
	gpumat::write_fs(fs, B);
}

void mlp::read(std::fstream &fs)
{
	gpumat::read_fs(fs, W);
	gpumat::read_fs(fs, B);
}

///**************************

void MlpOptim::init(const std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return;

	m_iteration = 0;

	m_mW.resize(_mlp.size());
	m_mb.resize(_mlp.size());

	m_vW.resize(_mlp.size());
	m_vb.resize(_mlp.size());

	sW.resize(_mlp.size());
	sB.resize(_mlp.size());

	for(size_t i = 0; i < _mlp.size(); i++){
		const gpumat::mlp& _mlpi = _mlp[i];
		m_mW[i].resize(_mlpi.W);
		m_vW[i].resize(_mlpi.W);
		m_mW[i].zeros();
		m_vW[i].zeros();

		m_mb[i].resize(_mlpi.B);
		m_vb[i].resize(_mlpi.B);
		m_mb[i].zeros();
		m_vb[i].zeros();
	}
	m_init_matB = true;
}

bool MlpOptim::pass(std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return false;

	if(!m_init_matB){
		throw new std::invalid_argument("MlpOptim::pass: not initialized");
	}

	m_iteration++;
	double sb1 = (1. / (1. - pow(m_betha1, m_iteration)));
	double sb2 = (1. / (1. - pow(m_betha2, m_iteration)));

	for(size_t i = 0; i < _mlp.size(); ++i){
		mlp& mlpi = _mlp[i];

		gpumat::add(m_mW[i], mlpi.gW, m_betha1, (1. - m_betha1));
		gpumat::add(m_mb[i], mlpi.gB, m_betha1, (1. - m_betha1));

		gpumat::elemwiseSqr(mlpi.gW, sW[i]);
		gpumat::elemwiseSqr(mlpi.gB, sB[i]);

		gpumat::add(m_vW[i], sW[i], m_betha2, (1. - m_betha2));
		gpumat::add(m_vb[i], sB[i], m_betha2, (1. - m_betha2));

		/// W = -alpha * (sb1 * mW / (sqrt(sb2 * vW) + eps))

//		gpumat::add(W[i], m_mW[i], 1, -m_alpha);
//		gpumat::add(b[i], m_mb[i], 1, -m_alpha);
		gpumat::sub_adamGrad(mlpi.W, m_mW[i], m_vW[i], m_alpha, sb1, sb2);
		gpumat::sub_adamGrad(mlpi.B, m_mb[i], m_vb[i], m_alpha, sb1, sb2);
	}
	return true;
}
