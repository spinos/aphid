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
	float * displayVertex();
	
	void setAlpha(float x);
	
protected:
    virtual void stepPhysics(float dt);	
private:
	void formPlane(float alpha);
	
private:
    float m_alpha;
    BaseBuffer * m_displayVertex;
	CUDABuffer * m_vertexBuffer;
	unsigned m_numTriIndices;
	unsigned * m_triIndices;
};