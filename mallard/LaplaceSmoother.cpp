/*
 *  LaplaceSmoother.cpp
 *  mallard
 *
 *  Created by jian zhang on 12/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "LaplaceSmoother.h"
#include <MeshTopology.h>
#include <VertexAdjacency.h>
LaplaceSmoother::LaplaceSmoother() : m_constraintIndices(0), m_constraintWeights(0){}
LaplaceSmoother::~LaplaceSmoother() 
{
	cleanup();
}

void LaplaceSmoother::cleanup()
{
	if(m_constraintIndices) delete[] m_constraintIndices;
	if(m_constraintWeights) delete[] m_constraintWeights;
	m_constraintIndices = 0;
	m_constraintWeights = 0;
}

unsigned LaplaceSmoother::numRows() const
{
	return m_numData + m_numConstraint;
}

void LaplaceSmoother::precompute(unsigned num, MeshTopology * topo, const std::vector<unsigned> & constraintIdx, const std::vector<float> & constraintWei)
{
	m_topology = topo;
	m_numData = num;
	m_numConstraint = constraintIdx.size();
	cleanup();
	m_constraintIndices = new unsigned[m_numConstraint];
	m_constraintWeights = new float[m_numConstraint];
	
	std::vector<unsigned>::const_iterator itidx = constraintIdx.begin();
	for(unsigned i = 0; itidx != constraintIdx.end(); ++itidx, ++i) m_constraintIndices[i] = *itidx;
	
	std::vector<float>::const_iterator itwei = constraintWei.begin();
	for(unsigned i = 0; itwei != constraintWei.end(); ++itwei, ++i) m_constraintWeights[i] = *itwei;
	
	int neighborIdx, lastNeighbor;
	LaplaceMatrixType L(m_numData + m_numConstraint, m_numData);
	L.startFill();
	for(int i = 0; i < (int)m_numData; i++) {
		VertexAdjacency & adj = topo->getAdjacency(i);
		lastNeighbor = -1;
		
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
			neighborIdx = neighbor->v->getIndex();
			if(neighborIdx > i && lastNeighbor < i) {
				L.fill(i, i) = 1.0f;
			}
			L.fill(i, neighborIdx) = -neighbor->weight;
			lastNeighbor = neighborIdx; 
		}
		if(lastNeighbor < i)
			L.fill(i, i) = 1.0f;
	}
	
	int irow = (int)m_numData;
	for(unsigned i = 0; i < m_numConstraint; i++) {
		L.fill(irow, m_constraintIndices[i]) = m_constraintWeights[i];
		irow++;
	}
	L.endFill();
	
	m_LT = L.transpose();
	LaplaceMatrixType m_M = m_LT * L;
	m_llt = Eigen::SparseLLT<LaplaceMatrixType>(m_M);
}

void LaplaceSmoother::computeRightHandSide(Vector3F * entry)
{
	m_delta[0].resize(numRows());
	m_delta[1].resize(numRows());
	m_delta[2].resize(numRows());
	
	for(unsigned i = 0; i < m_numData; i++) {
		VertexAdjacency & adj = m_topology->getAdjacency(i);
		Vector3F dif = adj.getDifferentialCoordinate();
		m_delta[0](i) = dif.x;
		m_delta[1](i) = dif.y;
		m_delta[2](i) = dif.z;
	}
	
	unsigned irow = m_numData;
	for(unsigned i = 0; i < m_numConstraint; i++) {
		Vector3F &p = entry[m_constraintIndices[i]];
		m_delta[0](irow) = p.x * m_constraintWeights[i];
		m_delta[1](irow) = p.y * m_constraintWeights[i];
		m_delta[2](irow) = p.z * m_constraintWeights[i];
		irow++;
	}
	
	m_delta[0] = m_LT * m_delta[0];
	m_delta[1] = m_LT * m_delta[1];
	m_delta[2] = m_LT * m_delta[2];
}

void LaplaceSmoother::solve(Vector3F * entry)
{
	computeRightHandSide(entry);

	m_llt.solveInPlace(m_delta[0]);
	m_llt.solveInPlace(m_delta[1]);
	m_llt.solveInPlace(m_delta[2]);
	
	for(unsigned i = 0; i < m_numData; i++) {
		entry[i].x = m_delta[0](i);
		entry[i].y = m_delta[1](i);
		entry[i].z = m_delta[2](i);
	}
}
