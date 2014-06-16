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
	m_margin = .1f;
}

MeshShape::~MeshShape() 
{
	if(m_indexVertexArrays) delete m_indexVertexArrays;
}

void MeshShape::setMargin(const float & x) { m_margin = x; }
const float MeshShape::margin() const { return m_margin; }

btBvhTriangleMeshShape* MeshShape::createCollisionShape()
{
    if (m_indexVertexArrays)
		delete m_indexVertexArrays;
	std::cout<<"num triangle: "<<numTri()<<"\n";
	std::cout<<"num vertex: "<<numVert()<<"\n";
	std::cout<<"sizeof btVector3: "<<sizeof(btVector3)<<"\n";
	m_indexVertexArrays = new btTriangleIndexVertexArray(numTri(), indices(), 3 * sizeof(int),
		numVert(),(btScalar*)&vertexPos()[0][0], sizeof(btVector3));
	
	const bool useQuantizedAabbCompression = true;
	const bool buildBvh = true;
	btBvhTriangleMeshShape* trimeshShape = new btBvhTriangleMeshShape(m_indexVertexArrays, useQuantizedAabbCompression, buildBvh);
    trimeshShape->setMargin(m_margin);
	PhysicsState::engine->addCollisionShape(trimeshShape);
	return trimeshShape;
}

}