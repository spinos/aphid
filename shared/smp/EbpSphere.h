/*
 *  EbpSphere.h
 *  
 *  energy based particle samples on a sphere of radius 10
 *
 *  Created by jian zhang on 7/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SMP_EBP_SPHERE_H
#define APH_SMP_EBP_SPHERE_H

#include "ebp.h"

namespace aphid {

class EbpSphere : public EbpGrid {

public:
	EbpSphere();
	virtual ~EbpSphere();
	
	int numSamples();
	
protected:

private:
};

}

#endif