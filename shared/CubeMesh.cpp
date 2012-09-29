/*
 *  CubeMesh.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "CubeMesh.h"

CubeMesh::CubeMesh()
{
	createVertices(8);
	createIndices(36);
	
	Vector3F * p = vertices();
	p[0] = Vector3F(1.f, 1.f, 1.f);
	p[1] = Vector3F(0.f, 1.f, 1.f);
	p[2] = Vector3F(0.f, 0.f, 1.f);
	p[3] = Vector3F(1.f, 0.f, 1.f);
	p[4] = Vector3F(1.f, 0.f, 0.f);
	p[5] = Vector3F(1.f, 1.f, 0.f);
	p[6] = Vector3F(0.f, 1.f, 0.f);
	p[7] = Vector3F(0.f, 0.f, 0.f);
	
	unsigned * idx = indices();
	
	idx[0] = 0;
	idx[1] = 1;
	idx[2] = 3;
	
	idx[3] = 2;
	idx[4] = 3;
	idx[5] = 0;
	
	idx[6] = 0;
	idx[7] = 3;
	idx[8] = 4;
	
	idx[9] = 4;
	idx[10] = 5;
	idx[11] = 0;
	
	idx[12] = 0;
	idx[13] = 5;
	idx[14] = 6;
	
	idx[15] = 6;
	idx[16] = 1;
	idx[17] = 0;
	
	idx[18] = 1;
	idx[19] = 6;
	idx[20] = 7;
	
	idx[21] = 7;
	idx[22] = 2;
	idx[23] = 1;
	
	idx[24] = 7;
	idx[25] = 4;
	idx[26] = 3;
	
	idx[27] = 3;
	idx[28] = 2;
	idx[29] = 7;
	
	idx[30] = 4;
	idx[31] = 7;
	idx[32] = 6;
	
	idx[33] = 6;
	idx[34] = 5;
	idx[35] = 4;
}

CubeMesh::~CubeMesh() {}
