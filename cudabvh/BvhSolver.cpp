/*
 *  BvhSolver.cpp
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtCore>
#include <CudaBase.h>
#include <CUDABuffer.h>
#include "BvhSolver.h"
#include "plane_implement.h"

BvhSolver::BvhSolver(QObject *parent) : BaseSolverThread(parent) 
{
	m_alpha = 0;
}

BvhSolver::~BvhSolver() {}

// i,j  i1,j  
// i,j1 i1,j1
//
// i,j  i1,j  
// i,j1
//
//		i1,j  
// i,j1 i1,j1

void BvhSolver::init()
{
	CudaBase::CheckCUDevice();
	CudaBase::SetDevice();
	qDebug()<<"solverinit";
	m_vertexBuffer = new CUDABuffer;
	m_vertexBuffer->create(numVertices() * 16);
	m_displayVertex = new BaseBuffer;
	m_displayVertex->create(numVertices() * 16);
	m_numTriIndices = 32 * 32 * 2 * 3;
	m_triIndices = new unsigned[m_numTriIndices];
	unsigned i, j;
	unsigned *ind = &m_triIndices[0];
	for(j=0; j < 32; j++) {
		for(i=0; i < 32; i++) {
			*ind = j * 33 + i;
			ind++;
			*ind = (j + 1) * 33 + i;
			ind++;
			*ind = j * 33 + i + 1;
			ind++;
			
			
			*ind = j * 33 + i + 1;
			ind++;
			*ind = (j + 1) * 33 + i;
			ind++;
			*ind = (j + 1) * 33 + i + 1;
			ind++;
			
		}
	}
}

void BvhSolver::stepPhysics(float dt)
{
	// qDebug()<<"step phy";
	formPlane(m_alpha);
}

void BvhSolver::formPlane(float alpha)
{
	// qDebug()<<"map";
	void *dptr = m_vertexBuffer->bufferOnDevice();
	// m_vertexBuffer->map(&dptr);
	
	wavePlane((float4 *)dptr, 32, 2.0, alpha);
	//qDebug()<<"out";
	// m_vertexBuffer->unmap();
	m_vertexBuffer->deviceToHost(m_displayVertex->data(), m_vertexBuffer->bufferSize());
}

// const unsigned BvhSolver::vertexBufferName() const { return m_vertexBuffer->bufferName(); }
const unsigned BvhSolver::numVertices() const { return (32 + 1 ) * (32 + 1); }

unsigned BvhSolver::getNumTriangleFaceVertices() const { return m_numTriIndices; }
unsigned * BvhSolver::getIndices() const { return m_triIndices; }
float * BvhSolver::displayVertex() { return (float *)m_displayVertex->data(); }
void BvhSolver::setAlpha(float x) { m_alpha = x; }