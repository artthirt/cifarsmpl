#include "nn.h"

namespace ct{

void get_cnv_sizes(const ct::Size sizeIn, const ct::Size szW, int stride, ct::Size &szA1, ct::Size &szA2)
{
	szA1.width		= (sizeIn.width - szW.width + 1) / stride;
	szA1.height		= (sizeIn.height - szW.height + 1) / stride;
	szA2			= ct::Size(szA1.width/2, szA1.height/2);
}

}
