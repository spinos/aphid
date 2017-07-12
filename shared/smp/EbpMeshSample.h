/*
 *  EbpMeshSample.h
 *  
 *  energy based particle samples on a triangle mesh along xy plane
 *
 *  Created by jian zhang on 7/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SMP_EBP_SPHERE_H
#define APH_SMP_EBP_SPHERE_H

#include "ebp.h"

namespace aphid {

class ATriangleMesh;

class EbpMeshSample : public EbpGrid {

public:
	EbpMeshSample();
	virtual ~EbpMeshSample();
	
	void sample(ATriangleMesh* msh);
	int numSamples();
	
protected:

private:
};

}

#endif