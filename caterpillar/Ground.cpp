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
	m_numVert = nv;
	return m_vertexPos;
}

int * Ground::createTriangles(const int & ntri)
{
	if(m_indices) delete[] m_indices;
	m_numTri = ntri;
	m_indices = new int[ntri * 3];
	return m_indices;
}

void Ground::create()
{
    if (m_indexVertexArrays)
		delete m_indexVertexArrays;
	
	m_indexVertexArrays = new btTriangleIndexVertexArray(m_numTri, m_indices, 3 * sizeof(int),
		m_numVert,(btScalar*)&m_vertexPos[0][0], sizeof(btVector3));
	
	const bool useQuantizedAabbCompression = true;
	const bool buildBvh = true;
	btBvhTriangleMeshShape* trimeshShape = new btBvhTriangleMeshShape(m_indexVertexArrays, useQuantizedAabbCompression, buildBvh);
    PhysicsState::engine->addCollisionShape(trimeshShape);
    
	// int id = PhysicsState::engine->numCollisionObjects();
    btTransform trans; trans.setIdentity();
	PhysicsState::engine->createRigidBody(trimeshShape, trans, 0.0);
}
}
