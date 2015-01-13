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
class CUDABuffer;
class BvhTriangleMesh {
public:
	BvhTriangleMesh();
	virtual ~BvhTriangleMesh();
	
	void createVertices(unsigned n);
	void createTriangles(unsigned n);
	
	void initOnDevice();
	
	const unsigned numVertices() const;
	const unsigned numTriangles() const;
	const unsigned numTriangleFaceVertices() const;
	
	void getVerticesOnDevice(BaseBuffer * dst);
	
	BaseBuffer * vertexBuffer();
	CUDABuffer * vertexBufferOnDevice();
	void * verticesOnDevice();
	Vector3F * vertices();
	unsigned * triangleIndices();
	
protected:

private:
	BaseBuffer * m_vertices;
	BaseBuffer * m_triangleIndices;
	CUDABuffer * m_verticesOnDevice;
	CUDABuffer * m_triangleIndicesOnDevice;
	unsigned m_numVertices, m_numTriangles;
};
#endif        //  #ifndef BVHTRIANGLEMESH_H
