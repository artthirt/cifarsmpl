#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

namespace ct{

enum etypefunction{
	LINEAR,
	RELU,
	SOFTMAX,
	SIGMOID,
	TANH
};

struct Size{
	Size(){
		width = height = 0;
	}
	Size(int w, int h){
		width = w;
		height = h;
	}
	int area() const{
		return width * height;
	}
	Size t(){
		return Size(height, width);
	}

	int width;
	int height;
};

struct ParamsMlp{
	ParamsMlp(){
		count = 0;
		this->prob = 1;
		this->lambda_l2 = 0;
	}
	ParamsMlp(int val, double prob, double lambda_l2 = 0){
		this->count = val;
		this->prob = prob;
		this->lambda_l2 = lambda_l2;
	}

	int count;
	double prob;
	double lambda_l2;
};

struct ParamsCnv{
	ParamsCnv(){
		size_w = 0;
		count_kernels = 0;
		pooling = true;
		prob = 1;
		lambda_l2 = 0.;
	}
	ParamsCnv(int size_w, int count_kernels, bool pooling, double prob, double lambda_l2){
		this->size_w = size_w;
		this->count_kernels = count_kernels;
		this->pooling = pooling;
		this->prob = prob;
		this->lambda_l2 = lambda_l2;
	}

	int size_w;
	int count_kernels;
	bool pooling;
	double prob;
	double lambda_l2;
};

}

#endif // COMMON_TYPES_H
