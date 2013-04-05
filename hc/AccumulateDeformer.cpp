/*
 *  AccumulateDeformer.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "AccumulateDeformer.h"

#include "VertexAdjacency.h"
#include "DeformationTarget.h"
#include "MeshLaplacian.h"

AccumulateDeformer::AccumulateDeformer() {}
AccumulateDeformer::~AccumulateDeformer() {}

void AccumulateDeformer::setMesh(BaseMesh * mesh)
{
	BaseDeformer::setMesh(mesh);
	
	printf("init accumulate deformer");
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_mesh);
	m_topology = msh->connectivity();
}

void AccumulateDeformer::setTargetAnalysis(DeformationTarget * analysis)
{
	m_targetAnalysis = analysis;
}

void AccumulateDeformer::precompute()
{
	int neighborIdx;
	m_L.resize(m_numVertices, m_numVertices);
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
			neighborIdx = neighbor->v->getIndex();
			m_L.insert(i, neighborIdx) = -neighbor->weight;
		}
		m_L.insert(i, i) = 1.0f;
	}
}

void AccumulateDeformer::prestep()
{
	LaplaceMatrixType L = m_L;
	for(unsigned i = 0; i < m_numVertices; i++) {
		L.coeffRef(i, i) = L.coeffRef(i, i) + .1f;
	}
	
	std::vector<unsigned> constrainIndices;
	m_targetAnalysis->genNonZeroIndices(constrainIndices);
	
	for(std::vector<unsigned>::const_iterator it = constrainIndices.begin(); it != constrainIndices.end(); ++it) {
		unsigned ic = *it;
		L.coeffRef(ic, ic) = L.coeffRef(ic, ic) + .8f;
	}
	
	LaplaceMatrixType LT = L.transpose();
	
	LaplaceMatrixType M = LT * L;
	m_llt.compute(M);
	
	m_delta[0].resize(m_numVertices);
	m_delta[1].resize(m_numVertices);
	m_delta[2].resize(m_numVertices);
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		Vector3F dif = adj.getDifferentialCoordinate();
		Vector3F worldP = m_targetAnalysis->restP(i);
		worldP *= .1f;
		m_delta[0](i) = dif.x + worldP.x;
		m_delta[1](i) = dif.y + worldP.y;
		m_delta[2](i) = dif.z + worldP.z;
	}
	
	for(std::vector<unsigned>::const_iterator it = constrainIndices.begin(); it != constrainIndices.end(); ++it) {
		unsigned ic = *it;
		VertexAdjacency & adj = m_topology[ic];
		Vector3F dif = adj.getDifferentialCoordinate();
		Matrix33F R = m_targetAnalysis->getR(ic);
		dif = R.transform(dif);
		Vector3F worldP = m_targetAnalysis->restP(ic) + m_targetAnalysis->getT(ic);
		worldP *= .9f;
		m_delta[0](ic) = dif.x + worldP.x;
		m_delta[1](ic) = dif.y + worldP.y;
		m_delta[2](ic) = dif.z + worldP.z;
	}
	
	m_delta[0] = LT * m_delta[0];
	m_delta[1] = LT * m_delta[1];
	m_delta[2] = LT * m_delta[2];
}

char AccumulateDeformer::solve()
{
	if(m_targetAnalysis->hasNoEffect()) {
		reset();
		return 0;
	}
	prestep();
	Eigen::VectorXf x[3];
	x[0] = m_llt.solve(m_delta[0]);
	x[1] = m_llt.solve(m_delta[1]);
	x[2] = m_llt.solve(m_delta[2]);
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_deformedV[i].x = x[0](i);
		m_deformedV[i].y = x[1](i);
		m_deformedV[i].z = x[2](i);
	}

	return 1;
}
