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
    m_trimeshShape = NULL;
}

Ground::~Ground() 
{
    if(m_indexVertexArrays) delete m_indexVertexArrays;
	if(m_trimeshShape) delete m_trimeshShape;
}

void Ground::create(const int & numTri, int * triangleIndices,
                    const int & numVert, float * verticesPositions)
{
    if (m_indexVertexArrays)
		delete m_indexVertexArrays;
	
	m_indexVertexArrays = new btTriangleIndexVertexArray(numTri, triangleIndices, 3*sizeof(int),
		numVert,(btScalar*)verticesPositions, sizeof(float));
	
	if(m_trimeshShape) delete m_trimeshShape;
	const bool useQuantizedAabbCompression = true;
	const bool buildBvh = true;
	m_trimeshShape = new btBvhTriangleMeshShape(m_indexVertexArrays, useQuantizedAabbCompression, buildBvh);
    int id = PhysicsState::engine->numCollisionObjects();
    btTransform trans; trans.setIdentity();
	PhysicsState::engine->createRigitBody(m_trimeshShape, trans, 0.0);
}
}
