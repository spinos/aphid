/*
 *  MeshShape.h
 *  wheeled
 *
 *  Created by jian zhang on 6/1/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Common.h>
namespace caterpillar {
class MeshShape {
public:
	MeshShape();
	virtual ~MeshShape();
	
protected:
	btVector3 * createVertexPos(const int & nv);
	int * createTriangles(const int & ntri);
	btBvhTriangleMeshShape* createCollisionShape();
	void setMargin(const float & x);
	
private:
	btTriangleIndexVertexArray* m_indexVertexArrays;
	btVector3 * m_vertexPos;
	int * m_indices;
	int m_numTri, m_numVert;
	float m_margin;
};
}