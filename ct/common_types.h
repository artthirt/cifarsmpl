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
	}
	ParamsMlp(int val, double prob){
		this->count = val;
		this->prob = prob;
	}

	int count;
	double prob;
};

struct ParamsCnv{
	ParamsCnv(){
		size_w = 0;
		count_kernels = 0;
		pooling = true;
		prob = 1;
	}
	ParamsCnv(int size_w, int count_kernels, bool pooling, double prob){
		this->size_w = size_w;
		this->count_kernels = count_kernels;
		this->pooling = pooling;
		this->prob = prob;
	}

	int size_w;
	int count_kernels;
	bool pooling;
	double prob;
};

}

#endif // COMMON_TYPES_H
