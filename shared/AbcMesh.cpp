/*
 *  AbcMesh.cpp
 *  AbcViewer
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "AbcMesh.h"
#include <iostream>
#include <cmath>

AbcMesh::AbcMesh(const char * filename) 
{
    std::cout<<"abc file "<<filename<<std::endl;
	createVertices(4);
	createIndices(6);
	
	Vector3F * p = vertices();
	
	unsigned * idx = indices();
	
	idx[0] = 0;
	idx[1] = 1;
	idx[2] = 2;
	idx[3] = 2;
	idx[4] = 3;
	idx[5] = 0;
	
	p[0].x = 0.f; p[0].y = 0.f; p[0].z = 2.f;
	p[1].x = 1.f; p[1].y = 0.f; p[1].z = 2.f;
	p[2].x = 1.f; p[2].y = 1.f; p[2].z = 2.f;
	p[3].x = 0.f; p[3].y = 1.f; p[3].z = 2.f;
}

AbcMesh::~AbcMesh() {}

