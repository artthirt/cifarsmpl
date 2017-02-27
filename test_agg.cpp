#include "test_agg.h"

#include <iostream>

#include "custom_types.h"
#include "gpumat.h"
#include "convnn_gpu.h"

#include "nn.h"

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

}
