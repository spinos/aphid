/*
 *  Hemisphere.h
 *  
 *
 *  Created by jian zhang on 2/9/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SMP_HEMISPHERE_H
#define APH_SMP_HEMISPHERE_H

namespace aphid {

class Vector3F;

namespace smp {

class Hemisphere {

public:

/// return point on hemisphere facing +z with cosine density
/// sampleP is sample point (phi,theta) within [0,1]
/// pDfW is probability density function weight
	static Vector3F SampleCosineHemisphere(const float* sampleP,
					float& pDfW);
	
};

}

}

#endif