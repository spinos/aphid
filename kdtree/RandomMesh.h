/*
 *  RandomMesh.h
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseMesh.h>

class RandomMesh : public BaseMesh {
public:
	RandomMesh(unsigned numFaces);
	virtual ~RandomMesh();
};