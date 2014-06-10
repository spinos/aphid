/*
 *  MeshShape.cpp
 *  wheeled
 *
 *  Created by jian zhang on 6/1/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "MeshShape.h"
#include <PhysicsState.h>
namespace caterpillar {
MeshShape::MeshShape() 
{
	m_indexVertexArrays = NULL;
	m_vertexPos = NULL;
	m_indices = NULL;
	m_margin = .1f;
}

MeshShape::~MeshShape() 
{
	if(m_indexVertexArrays) delete m_indexVertexArrays;
	if(m_vertexPos) delete[] m_vertexPos;
	if(m_indices) delete[] m_indices;
}

btVector3 * MeshShape::createVertexPos(const int & nv)
{
	if(m_vertexPos) delete[] m_vertexPos;
	m_vertexPos = new btVector3[nv];
	m_numVert = nv;
	return m_vertexPos;
}

int * MeshShape::createTriangles(const int & ntri)
{
	if(m_indices) delete[] m_indices;
	m_numTri = ntri;
	m_indices = new int[ntri * 3];
	return m_indices;
}

void MeshShape::setMargin(const float & x) { m_margin = x; }

btBvhTriangleMeshShape* MeshShape::createCollisionShape()
{
    if (m_indexVertexArrays)
		delete m_indexVertexArrays;
	std::cout<<"num triangle: "<<m_numTri<<"\n";
	std::cout<<"num vertex: "<<m_numVert<<"\n";
	std::cout<<"sizeof btVector3: "<<sizeof(btVector3)<<"\n";
	m_indexVertexArrays = new btTriangleIndexVertexArray(m_numTri, m_indices, 3 * sizeof(int),
		m_numVert,(btScalar*)&m_vertexPos[0][0], sizeof(btVector3));
	
	const bool useQuantizedAabbCompression = true;
	const bool buildBvh = true;
	btBvhTriangleMeshShape* trimeshShape = new btBvhTriangleMeshShape(m_indexVertexArrays, useQuantizedAabbCompression, buildBvh);
    trimeshShape->setMargin(m_margin);
	PhysicsState::engine->addCollisionShape(trimeshShape);
	return trimeshShape;
}

}