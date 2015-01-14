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

SimpleMesh::SimpleMesh() { m_alpha = 0; }
SimpleMesh::~SimpleMesh() {}

void SimpleMesh::setAlpha(float x) 
{ m_alpha = x; }

const float SimpleMesh::alpha() const
{ return m_alpha; }

void SimpleMesh::setPlaneUDim(unsigned x)
{ m_planeUDim = x; }

void SimpleMesh::update()
{
	void *dptr = verticesOnDevice();
	wavePlane((float3 *)dptr, m_planeUDim, 2.0, m_alpha);
}