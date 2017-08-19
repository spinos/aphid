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

BillboardMesh::BillboardMesh()
{}

BillboardMesh::~BillboardMesh()
{}

void BillboardMesh::setBillboardSize(float w, float h, int nu, int addNv)
{
	int nv = 4 + h / w * nu + addNv;
	if(nv < 4)
		nv = 4;
	
	float du = w / (float)nu;
	float dv = h / (float)nv; 
	
	createGrid(nu, nv, du, dv);
	
	int np = numPoints();
	Vector3F * p = points();
	
	const float hw = w * .5f;
	for(int i = 0; i < np; i++) {
		p[i].x -= hw;
	}
}



}