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
	createVertices(19);
	createIndices(18 * 3);
	
	Vector3F * p = vertices();
	
	unsigned * idx = indices();

	float ang;
	const float da = 3.14159269f * 2.f / 18.f;
	for(int i = 0; i < 18; i++) {
		ang = da * i;
		p[i].x = cos(ang);
		p[i].y = sin(ang);
		p[i].z = 0.f;
	}
	
	p[18].setZero();
	
	
	for(int i = 0; i < 18; i++) {
		idx[i * 3] = 18;
		idx[i * 3 + 1] = i;
		idx[i * 3 + 2] = (i + 1) % 18;
	}
}

DiscMesh::~DiscMesh()
{

}