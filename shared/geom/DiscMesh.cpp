/*
 *  DiscMesh.cpp
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "DiscMesh.h"
#include <cmath>
namespace aphid {
    
DiscMesh::DiscMesh(int nseg)
{
    create(nseg+1, nseg);
	
	Vector3F * p = points();
	unsigned * idx = indices();

	float ang;
	const float da = 3.14159269f * 2.f / (float)nseg;
	for(int i = 0; i < nseg; i++) {
		ang = da * i;
		p[i].set( cos(ang), sin(ang), 0.f );
	}
	
	p[nseg].setZero();
	
	for(int i = 0; i < nseg; i++) {
		idx[i * 3] = nseg;
		idx[i * 3 + 1] = i;
		idx[i * 3 + 2] = (i + 1) % nseg;
	}
}

DiscMesh::~DiscMesh()
{}

}