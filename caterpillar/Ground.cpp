/*
 *  Ground.cpp
 *  caterpillar
 *
 *  Created by jian zhang on 5/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Ground.h"
#include <DynamicsSolver.h>
#include "PhysicsState.h"
namespace caterpillar {

Ground::Ground() 
{
    m_indexVertexArrays = NULL;
	m_vertexPos = NULL;
	m_indices = NULL;
}

Ground::~Ground() 
{
    if(m_indexVertexArrays) delete m_indexVertexArrays;
	if(m_vertexPos) delete[] m_vertexPos;
	if(m_indices) delete[] m_indices;
}

btVector3 * Ground::createVertexPos(const int & nv)
{
	if(m_vertexPos) delete[] m_vertexPos;
	m_vertexPos = new btVector3[nv];
	return m_vertexPos;
}

int * Ground::createVertexIndex(const int & ni)
{
	if(m_indices) delete[] m_indices;
	m_indices = new int[ni];
	return m_indices;
}

void Ground::create(const int & numTri, int * triangleIndices,
                    const int & numVert, btVector3 * verticesPositions)
{
    if (m_indexVertexArrays)
		delete m_indexVertexArrays;
	
	m_indexVertexArrays = new btTriangleIndexVertexArray(numTri, triangleIndices, 3 * sizeof(int),
		numVert,(btScalar*)&verticesPositions[0][0], sizeof(btVector3));
	
	const bool useQuantizedAabbCompression = true;
	const bool buildBvh = true;
	btBvhTriangleMeshShape* trimeshShape = new btBvhTriangleMeshShape(m_indexVertexArrays, useQuantizedAabbCompression, buildBvh);
    PhysicsState::engine->addCollisionShape(trimeshShape);
    
	int id = PhysicsState::engine->numCollisionObjects();
    btTransform trans; trans.setIdentity();
	PhysicsState::engine->createRigitBody(trimeshShape, trans, 0.0);
}
}
