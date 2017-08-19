/*
 *  BladeMesh.h
 *  
 *  m-by-n grid with merged tip
 *  m # v segments n # columns must be even
 *  nv <- (m+1) * n + n / 2
 *  n profiles m + 2 vertices each from root to tip
 *  opposite profiles share last vertex on the top 
 *
 *  Created by jian zhang on 8/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_BLADE_MESH_H
#define APH_BLADE_MESH_H

#include "LoftMesh.h"

namespace aphid {

class BladeMesh : public LoftMesh {
	
public:
	BladeMesh();
	virtual ~BladeMesh();
/// center line (rib) with has no effect if n < 4
	virtual void createBlade(const float& width, const float& height,
					const float& ribWidth, const float& tipHeight,
					const int& m, const int& n);
	
protected:
	
private:
};

}
#endif
