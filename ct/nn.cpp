#include "nn.h"

namespace ct{

void get_cnv_sizes(const ct::Size sizeIn, const ct::Size szW, int stride, ct::Size &szA1, ct::Size &szA2)
{
	szA1.width		= (sizeIn.width - szW.width) / stride + 1;
	szA1.height		= (sizeIn.height - szW.height) / stride + 1;
	szA2			= ct::Size(szA1.width/2, szA1.height/2);

	if((szA2.width % 2) == 1) szA2.width += 1;
	if((szA2.height % 2) == 1) szA2.height += 1;
}

}
