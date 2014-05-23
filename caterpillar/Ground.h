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
	void create(const int & numTri, int * triangleIndices,
				const int & numVert, btVector3 * verticesPositions);
				
	btVector3 * createVertexPos(const int & nv);
	int * createVertexIndex(const int & ni);
private:
    btTriangleIndexVertexArray* m_indexVertexArrays;
	btVector3 * m_vertexPos;
	int * m_indices;
};

}
