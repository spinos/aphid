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

class AbcMesh : public BaseMesh {
public:
	AbcMesh(const char * filename);
	virtual ~AbcMesh();
};