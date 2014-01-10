/*
 *  DiscMesh.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/11/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "DiscMesh.h"

DiscMesh::DiscMesh()
{
	createVertices(37);
	createIndices(36 * 3);
	
	Vector3F * p = vertices();
	
	unsigned * idx = indices();

	float ang;
	const float da = 3.14159269f * 2.f / 36.f;
	for(int i = 0; i < 36; i++) {
		ang = da * i;
		p[i].x = cos(ang);
		p[i].y = sin(ang);
		p[i].z = 0.f;
	}
	
	p[36].setZero();
	
	
	for(int i = 0; i < 36; i++) {
		idx[i * 3] = 36;
		idx[i * 3 + 1] = i;
		idx[i * 3 + 2] = (i + 1) % 36;
	}
}

DiscMesh::~DiscMesh()
{

}