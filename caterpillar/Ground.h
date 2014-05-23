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
namespace caterpillar {
class Ground {
public:
	Ground();
	virtual ~Ground();
	void create(const int & numTri, int * triangleIndices,
                    const int & numVert, float * verticesPositions);
private:
    btTriangleIndexVertexArray* m_indexVertexArrays;
};

}
