#ifndef BVHTRIANGLEMESH_H
#define BVHTRIANGLEMESH_H

/*
 *  BvhTriangleMesh.h
 *  cudabvh
 *
 *  Created by jian zhang on 1/14/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <AllMath.h>
class BaseBuffer;

class BvhTriangleMesh {
public:
	BvhTriangleMesh();
	virtual ~BvhTriangleMesh();
	
	void createVertices(unsigned n);
	void createTriangles(unsigned n);
	
	const unsigned numVertices() const;
	const unsigned numTriangles() const;
	const unsigned numTriangleFaceVertices() const;
	
	BaseBuffer * vertexBuffer();
	Vector3F * vertices();
	unsigned * triangleIndices();
	
protected:

private:
	BaseBuffer * m_vertices;
	BaseBuffer * m_triangleIndices;
	unsigned m_numVertices, m_numTriangles;
};
#endif        //  #ifndef BVHTRIANGLEMESH_H
