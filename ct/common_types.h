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

struct ParamsCommon{
	double prob;
	double lambda_l2;
	int count;

	ParamsCommon(){
		prob = 1;
		lambda_l2 = 0;
		count = 0;
	}
};

struct ParamsMlp: public ParamsCommon{
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
};

struct ParamsCnv: public ParamsCommon{
	ParamsCnv(){
		size_w = 0;
		count = 0;
		pooling = true;
		prob = 1;
		lambda_l2 = 0.;
	}
	ParamsCnv(int size_w, int count_kernels, bool pooling, double prob, double lambda_l2){
		this->size_w = size_w;
		this->count = count_kernels;
		this->pooling = pooling;
		this->prob = prob;
		this->lambda_l2 = lambda_l2;
	}

	int size_w;
	bool pooling;
};

}

#endif // COMMON_TYPES_H
