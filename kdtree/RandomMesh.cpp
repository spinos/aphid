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
#include <cmath>
RandomMesh::RandomMesh(unsigned numFaces, const Vector3F & center, const float & size, int type) 
{
	createVertices(numFaces * 3);
	createIndices(numFaces * 3);
	
	Vector3F * p = vertices();
	
	unsigned * idx = indices();
	
	float rx, ry, rz, r, phi, theta;
	for(unsigned i = 0; i < numFaces; i++) {
		idx[i * 3] = i * 3;
		idx[i * 3 + 1] = i * 3 + 1;
		idx[i * 3 + 2] = i * 3 + 2;

		if(type == 0) {
			rx = (float(random()%694) / 694.f - 0.5f) * 1.4f;
			ry = (float(random()%594) / 594.f - 0.5f) * 1.4f;
			rz = (float(random()%794) / 794.f - 0.5f) * 1.4f;
		}
		else {
			phi = ((float)(rand() % 25391)) / 25391.f * 2.f * 3.14f;
			theta = ((float)(rand() % 24331)) / 24331.f * 3.14f;
			r = ((float)(rand() % 24091)) / 24091.f * .2f + 0.8f;
			rx = sin(theta) * cos(phi);
			ry = sin(theta) * sin(phi);
			rz = cos(theta);
		}
		p[i * 3] = center + Vector3F(rx * size, ry * size, rz * size);	
		
		rx = float(random()%294) / 294.f - 0.5f;
		ry = float(random()%594) / 594.f - 0.5f;
		rz = float(random()%794) / 794.f - 0.5f;
		
		p[i * 3 + 1] = p[i * 3] + Vector3F(rx, ry, rz);
		
		rx = float(random()%394) / 394.f - 0.5f;
		ry = float(random()%594) / 594.f - 0.5f;
		rz = float(random()%794) / 794.f - 0.5f;
		
		p[i * 3 + 2] = p[i * 3] + Vector3F(rx, ry, rz);
	}
}

RandomMesh::~RandomMesh() {}

