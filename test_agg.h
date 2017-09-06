#ifndef TEST_AGG_H
#define TEST_AGG_H


class test_agg
{
public:
	test_agg();

	void test_hconcat();
	void test_im2col();
	void test_conv();
	void test_conv_gpu();
	void test_file();
	void test_norm();
	void test_back();
	void test_conv2();
};

#endif // TEST_AGG_H
