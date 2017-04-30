/*
 *  PyramidMesh.cpp
 *  fit
 *
 *  Created by jian zhang on 5/3/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "PyramidMesh.h"
namespace aphid {

PyramidMesh::PyramidMesh()
{
	createVertices(5);
	createIndices(18);
	
	Vector3F * p = vertices();
	
	unsigned * idx = indices();
	
	p[0].setZero();
	p[1] = Vector3F(-.5f, -1.f,  .5f);
	p[2] = Vector3F( .5f, -1.f,  .5f);
	p[3] = Vector3F( .5f, -1.f, -.5f);
	p[4] = Vector3F(-.5f, -1.f, -.5f);
	
	idx[0] = 0; idx[1] = 1; idx[2] = 2;
	idx[3] = 0; idx[4] = 2; idx[5] = 3;
	idx[6] = 0; idx[7] = 3; idx[8] = 4;
	idx[9] = 0; idx[10] = 4; idx[11] = 1;
	idx[12] = 1; idx[13] = 3; idx[14] = 2;
	idx[15] = 1; idx[16] = 4; idx[17] = 3;
}

PyramidMesh::~PyramidMesh()
{

}

}
