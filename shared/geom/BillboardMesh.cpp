/*
 *  BillboardMesh.cpp
 *
 *  grid mesh with width, height, nu = 1 
 *  centered at (0, height/2) along xy plane facing +z
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "BillboardMesh.h"

namespace aphid {

BillboardMesh::BillboardMesh(float w, float h)
{
	setBillboardSize(w, h);
}

BillboardMesh::~BillboardMesh()
{}

void BillboardMesh::setBillboardSize(float w, float h)
{
	int nu = 1;
	int nv = 3 + h / w;
	
	float dv = h / (float)nv; 
	
	createGrid(nu, nv, w, dv);
	
	int np = numPoints();
	Vector3F * p = points();
	
	const float hw = w * .5f;
	for(int i = 0; i < np; i++) {
		p[i].x -= hw;
	}
}



}