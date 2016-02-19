/*
 *  RandomMesh.h
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <ATriangleMesh.h>

class RandomMesh : public ATriangleMesh {
public:
	RandomMesh(unsigned numFaces, const Vector3F & center, const float & size, int type);
	virtual ~RandomMesh();
};