/*
 *  BvhSolver.h
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BaseSolverThread.h>
class BaseBuffer;
class CUDABuffer;
struct EdgeContact;
class BvhSolver : public BaseSolverThread
{
public:
	BvhSolver(QObject *parent = 0);
	virtual ~BvhSolver();
	
	void init();
	// const unsigned vertexBufferName() const;
	const unsigned numVertices() const;
	
	unsigned getNumTriangleFaceVertices() const;
	unsigned * getIndices() const;
	unsigned numEdges() const;
	float * displayVertex();
	EdgeContact * edgeContacts();
	void setAlpha(float x);
	
protected:
    virtual void stepPhysics(float dt);	
private:
	void formPlane(float alpha);
	
private:
    float m_alpha;
    BaseBuffer * m_displayVertex;
	CUDABuffer * m_vertexBuffer;
	unsigned m_numTriIndices, m_numTriangles, m_numEdges;
	unsigned * m_triIndices;
	BaseBuffer * m_edges;
};