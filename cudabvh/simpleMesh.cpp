/*
 *  simpleMesh.cpp
 *  cudabvh
 *
 *  Created by jian zhang on 1/14/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "simpleMesh.h"
#include "plane_implement.h"

#define IDIM 125
#define IDIM1 126

SimpleMesh::SimpleMesh() 
{ 
	m_alpha = 0; 
	createVertices(IDIM1 * IDIM1);
	createTriangles(IDIM * IDIM * 2);
	
// i,j  i1,j  
// i,j1 i1,j1
//
// i,j  i1,j  
// i,j1
//		i1,j  
// i,j1 i1,j1

	unsigned i, j, i1, j1;
	unsigned *ind = triangleIndices();
	for(j=0; j < IDIM; j++) {
	    j1 = j + 1;
		for(i=0; i < IDIM; i++) {
		    i1 = i + 1;
			*ind = j * IDIM1 + i;
			ind++;
			*ind = j1 * IDIM1 + i;
			ind++;
			*ind = j * IDIM1 + i1;
			ind++;

			*ind = j * IDIM1 + i1;
			ind++;
			*ind = j1 * IDIM1 + i;
			ind++;
			*ind = j1 * IDIM1 + i1;
			ind++;
		}
	}
}

SimpleMesh::~SimpleMesh() {}

void SimpleMesh::setAlpha(float x) 
{ m_alpha = x; }

const float SimpleMesh::alpha() const
{ return m_alpha; }

void SimpleMesh::update()
{
	void *dptr = verticesOnDevice();
	wavePlane((float3 *)dptr, IDIM, 2.0, m_alpha);
}