/*
 *  RandomMesh.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "RandomMesh.h"
#include <iostream>

RandomMesh::RandomMesh(unsigned numFaces) 
{
	createVertices(numFaces * 3);
	createIndices(numFaces * 3);
	
	Vector3F * p = vertices();
	
	unsigned * idx = indices();
	
	float rx, ry, rz;
	for(unsigned i = 0; i < numFaces; i++) {
		idx[i * 3] = i * 3;
		idx[i * 3 + 1] = i * 3 + 1;
		idx[i * 3 + 2] = i * 3 + 2;
		
		rx = float(random()%694) / 694.f;
		ry = float(random()%594) / 594.f;
		rz = float(random()%794) / 794.f;
		
		p[i * 3] = Vector3F(rx * 29.f + 1.f, ry * 29.f + 1.f, rz * 29.f + 1.f);
		
		rx = float(random()%294) / 294.f - 0.5f;
		ry = float(random()%594) / 594.f - 0.5f;
		rz = float(random()%794) / 794.f - 0.5f;
		
		p[i * 3 + 1] = p[i * 3] + Vector3F(rx, ry, rz) * 2.f;
		
		rx = float(random()%394) / 394.f - 0.5f;
		ry = float(random()%594) / 594.f - 0.5f;
		rz = float(random()%794) / 794.f - 0.5f;
		
		p[i * 3 + 2] = p[i * 3] + Vector3F(rx, ry, rz) * 2.f;
	}
}

RandomMesh::~RandomMesh() {}

