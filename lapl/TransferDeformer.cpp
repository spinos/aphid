/*
 *  TransferDeformer.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "TransferDeformer.h"

#include "VertexAdjacency.h"
#include "MeshLaplacian.h"

TransferDeformer::TransferDeformer() {}
TransferDeformer::~TransferDeformer() {}

void TransferDeformer::setMesh(BaseMesh * mesh)
{
	BaseDeformer::setMesh(mesh);
	
	printf("init laplace deformer");
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_mesh);
	m_topology = msh->connectivity();
}

void TransferDeformer::precompute(std::vector<Anchor *> & anchors)
{
	unsigned char *isAnchor = new unsigned char[m_numVertices];
	for(int i = 0; i < (int)m_numVertices; i++) isAnchor[i] = 0;
	
	m_anchorPoints.clear();
	for(std::vector<Anchor *>::iterator it = anchors.begin(); it != anchors.end(); ++it) {
		unsigned idx;
		for(Anchor::AnchorPoint * ap = (*it)->firstPoint(idx); (*it)->hasPoint(); ap = (*it)->nextPoint(idx)) {
			m_anchorPoints[idx] = ap;
			isAnchor[idx] = 1;
		}
	}
	
	int neighborIdx, lastNeighbor;
	LaplaceMatrixType L(m_numVertices + numAnchorPoints(), m_numVertices);
	L.startFill();
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
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
	
	int irow = (int)m_numVertices;
	std::map<unsigned, Anchor::AnchorPoint *>::iterator it;
	for(it = m_anchorPoints.begin(); it != m_anchorPoints.end(); ++it) {
		unsigned idx = (*it).first;
		L.fill(irow, idx) = 1.0f;
		irow++;
	}
	L.endFill();
	
	m_LT = L.transpose();
	LaplaceMatrixType m_M = m_LT * L;
	//std::cout << "M \n" << m_M << std::endl;
	m_llt = Eigen::SparseLLT<LaplaceMatrixType>(m_M);
	
	delete[] isAnchor;
}

unsigned TransferDeformer::numAnchorPoints() const
{
	return m_anchorPoints.size();
}

void TransferDeformer::prestep()
{
	m_delta[0].resize(m_numVertices + numAnchorPoints());
	m_delta[1].resize(m_numVertices + numAnchorPoints());
	m_delta[2].resize(m_numVertices + numAnchorPoints());
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];

		m_delta[0](i) = 0.f;
		m_delta[1](i) = 0.f;
		m_delta[2](i) = 0.f;
		
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
		    Vector3F pij = *(adj.m_v) - *(neighbor->v->m_v);
			m_delta[0](i) += pij.x * neighbor->weight;
			m_delta[1](i) += pij.y * neighbor->weight;
			m_delta[2](i) += pij.z * neighbor->weight;
		}
	}
	int irow = (int)m_numVertices;
	std::map<unsigned, Anchor::AnchorPoint *>::iterator it;
	for(it = m_anchorPoints.begin(); it != m_anchorPoints.end(); ++it) {
		Anchor::AnchorPoint *ap = (*it).second;
		m_delta[0](irow) = ap->worldP.x;
		m_delta[1](irow) = ap->worldP.y;
		m_delta[2](irow) = ap->worldP.z;
		irow++;
	}
	
	m_delta[0] = m_LT * m_delta[0];
	m_delta[1] = m_LT * m_delta[1];
	m_delta[2] = m_LT * m_delta[2];
}

char TransferDeformer::solve()
{
	prestep();
	m_llt.solveInPlace(m_delta[0]);
	m_llt.solveInPlace(m_delta[1]);
	m_llt.solveInPlace(m_delta[2]);
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_deformedV[i].x = m_delta[0](i);
		m_deformedV[i].y = m_delta[1](i);
		m_deformedV[i].z = m_delta[2](i);
	}

	return 1;
}
