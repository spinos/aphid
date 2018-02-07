/*
 *  Hemisphere.cpp
 *  
 *
 *  Created by jian zhang on 2/9/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "Hemisphere.h"
#include <math/Vector3F.h>
#include <math/miscfuncs.h>

namespace aphid {

namespace smp {

Vector3F Hemisphere::SampleCosineHemisphere(const float* sampleP,
					float& pDfW)
{
	const float term1 = 2.f * PIF * sampleP[0];
    const float term2 = std::sqrt(1.f - sampleP[1]);

    Vector3F r(std::cos(term1) * term2,
        std::sin(term1) * term2,
        std::sqrt(sampleP[1]));

    pDfW = r.z * ONE_OVER_PIF;

    return r;
}

}

}