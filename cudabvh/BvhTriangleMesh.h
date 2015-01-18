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
#include "CollisionObject.h"
class BaseBuffer;
class CUDABuffer;
class BvhTriangleMesh : public CollisionObject {
public:
	BvhTriangleMesh();
	virtual ~BvhTriangleMesh();
	
	void createVertices(unsigned n);
	void createTriangles(unsigned n);
	
	virtual void initOnDevice();
	virtual void update();
	
	const unsigned numVertices() const;
	const unsigned numTriangles() const;
	const unsigned numTriangleFaceVertices() const;
	
	void getVerticesOnDevice(BaseBuffer * dst);
	void deviceToHost();
	
	BaseBuffer * vertexBuffer();
	CUDABuffer * vertexBufferOnDevice();
	void * verticesOnDevice();
	void * triangleIndicesOnDevice();
	Vector3F * vertices();
	unsigned * triangleIndices();
	
protected:
    
private:
    void formTriangleAabbs();
    void combineAabbFirst();

private:
	BaseBuffer * m_vertices;
	BaseBuffer * m_triangleIndices;
	CUDABuffer * m_verticesOnDevice;
	CUDABuffer * m_triangleIndicesOnDevice;
	unsigned m_numVertices, m_numTriangles;
};
#endif        //  #ifndef BVHTRIANGLEMESH_H
