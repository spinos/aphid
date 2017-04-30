/*
 *  Ground.h
 *  caterpillar
 *
 *  Created by jian zhang on 5/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
class btTriangleIndexVertexArray;
class btBvhTriangleMeshShape;
class btVector3;
namespace caterpillar {
class Ground {
public:
	Ground();
	virtual ~Ground();
		
protected:			
	btVector3 * createVertexPos(const int & nv);
	int * createTriangles(const int & ntri);
	void create();
	void setMargin(const float & x);
	void setFriction(const float & x);
	const float scaling() const;
private:
    btTriangleIndexVertexArray* m_indexVertexArrays;
	btVector3 * m_vertexPos;
	int * m_indices;
	int m_numTri, m_numVert;
	float m_friction, m_margin;
};

}
