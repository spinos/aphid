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
#include "createBvh_implement.h"

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
	m_numTriangles = 32 * 32 * 2;
	m_numTriIndices = m_numTriangles * 3;
	m_triIndices = new unsigned[m_numTriIndices];
	unsigned i, j, i1, j1;
	unsigned *ind = &m_triIndices[0];
	for(j=0; j < 32; j++) {
	    j1 = j + 1;
		for(i=0; i < 32; i++) {
		    i1 = i + 1;
			*ind = j * 33 + i;
			ind++;
			*ind = j1 * 33 + i;
			ind++;
			*ind = j * 33 + i1;
			ind++;

			*ind = j * 33 + i1;
			ind++;
			*ind = j1 * 33 + i;
			ind++;
			*ind = j1 * 33 + i1;
			ind++;
		}
	}
	
	m_numEdges = 32 * (32 + 1) + 32 * (32 + 1) + 32 * 32;
	m_edges = new BaseBuffer;
	m_edges->create(m_numEdges * sizeof(EdgeContact));
	EdgeContact * edge = (EdgeContact *)(&m_edges->data()[0]);
	
	for(j=0; j < 33; j++) {
	    j1 = j + 1;
		for(i=0; i < 32; i++) {
		    i1 = i + 1;
		    if(j==0) {
		        edge->v[0] = i1;
		        edge->v[1] = i;
		        edge->v[2] = 33 + i;
		        edge->v[3] = MAX_INDEX;
		    }
		    else if(j==32) {
		        edge->v[0] = j * 33 + i;
		        edge->v[1] = j * 33 + i1;
		        edge->v[2] = (j - 1) * 33 + i1;
		        edge->v[3] = MAX_INDEX;
		    }
		    else {
		        edge->v[0] = j * 33 + i;
		        edge->v[1] = j * 33 + i1;
		        edge->v[2] = (j - 1) * 33 + i1;
		        edge->v[3] = j1 * 33 + i;
		    }
		    edge++;
		}
	}
	
	for(j=0; j < 32; j++) {
	    j1 = j + 1;
		for(i=0; i < 33; i++) {
		    i1 = i + 1;
		    if(i==0) {
		        edge->v[0] = j * 33 + i;
		        edge->v[1] = j1 * 33 + i;
		        edge->v[2] = j * 33 + i1;
		        edge->v[3] = MAX_INDEX;
		    }
		    else if(i==32) {
		        edge->v[0] = j1 * 33 + i;
		        edge->v[1] = j * 33 + i;
		        edge->v[2] = j1 * 33 + i - 1;
		        edge->v[3] = MAX_INDEX;
		    }
		    else {
		        edge->v[0] = j1 * 33 + i;
		        edge->v[1] = j * 33 + i;
		        edge->v[2] = j1 * 33 + i - 1;
		        edge->v[3] = j * 33 + i1;
		    }
		    edge++;
		}
	}
	
	for(j=0; j < 32; j++) {
	    j1 = j + 1;
		for(i=0; i < 32; i++) {
		    i1 = i + 1;
		    edge->v[0] = j1 * 33 + i;
		    edge->v[1] = j * 33 + i1;
		    edge->v[2] = j  * 33 + i;
		    edge->v[3] = j1 * 33 + i1;
		}
		edge++;
	}
	
	m_edgeContactIndices = new CUDABuffer;
	m_edgeContactIndices->create(m_numEdges * sizeof(EdgeContact));
	m_edgeContactIndices->hostToDevice(m_edges->data(), m_edgeContactIndices->bufferSize());
	
	m_allAabbs = new CUDABuffer;
	m_allAabbs->create(m_numEdges * sizeof(Aabb));

#ifdef BVHSOLVER_DBG_DRAW	
	m_displayAabbs = new BaseBuffer;
	m_displayAabbs->create(m_numEdges * sizeof(Aabb));
#endif

	qDebug()<<"num triangles "<<m_numTriangles;
	qDebug()<<"num edges "<<m_numTriangles;
}

void BvhSolver::stepPhysics(float dt)
{
	formPlane(m_alpha);
	formAabbs();
}

void BvhSolver::formPlane(float alpha)
{
	void *dptr = m_vertexBuffer->bufferOnDevice();
	wavePlane((float4 *)dptr, 32, 2.0, alpha);
	m_vertexBuffer->deviceToHost(m_displayVertex->data(), m_vertexBuffer->bufferSize());
}

void BvhSolver::formAabbs()
{
    void * cvs = m_vertexBuffer->bufferOnDevice();
    void * edges = m_edgeContactIndices->bufferOnDevice();
    void * dst = m_allAabbs->bufferOnDevice();
    bvhCalculateAabbs((Aabb *)dst, (float4 *)cvs, (EdgeContact *)edges, m_numEdges, numVertices());
    
#ifdef BVHSOLVER_DBG_DRAW
    m_allAabbs->deviceToHost(m_displayAabbs->data(), m_allAabbs->bufferSize());
#endif

}

const unsigned BvhSolver::numVertices() const { return (32 + 1 ) * (32 + 1); }

unsigned BvhSolver::getNumTriangleFaceVertices() const { return m_numTriIndices; }
unsigned * BvhSolver::getIndices() const { return m_triIndices; }
float * BvhSolver::displayVertex() { return (float *)m_displayVertex->data(); }
EdgeContact * BvhSolver::edgeContacts() { return (EdgeContact *)m_edges->data(); }
unsigned BvhSolver::numEdges() const { return m_numEdges; }

#ifdef BVHSOLVER_DBG_DRAW
Aabb * BvhSolver::displayAabbs() { return (Aabb *)m_displayAabbs->data(); }
#endif

void BvhSolver::setAlpha(float x) { m_alpha = x; }