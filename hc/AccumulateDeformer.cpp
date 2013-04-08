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
		m_L.insert(i, i) = 1.3f;
	}
	
	m_delta[0].resize(m_numVertices);
	m_delta[1].resize(m_numVertices);
	m_delta[2].resize(m_numVertices);
	
	Vector3F *v = m_mesh->getVertices();
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		Vector3F dif = adj.getDifferentialCoordinate();
		Vector3F worldP = v[i];
		worldP *= .3f;
		m_delta[0](i) = dif.x + worldP.x;
		m_delta[1](i) = dif.y + worldP.y;
		m_delta[2](i) = dif.z + worldP.z;
	}
}

void AccumulateDeformer::prestep(Eigen::VectorXf b[])
{
	LaplaceMatrixType L = m_L;
	
	std::vector<unsigned> constrainIndices;
	m_targetAnalysis->genNonZeroIndices(constrainIndices);
	
	for(std::vector<unsigned>::const_iterator it = constrainIndices.begin(); it != constrainIndices.end(); ++it) {
		unsigned ic = *it;
		L.coeffRef(ic, ic) = L.coeffRef(ic, ic) + .7f;
	}
	
	LaplaceMatrixType LT = L.transpose();
	
	LaplaceMatrixType M = LT * L;
	m_llt.compute(M);
	
	b[0] = m_delta[0];
	b[1] = m_delta[1];
	b[2] = m_delta[2];
	
	for(std::vector<unsigned>::const_iterator it = constrainIndices.begin(); it != constrainIndices.end(); ++it) {
		unsigned ic = *it;
		VertexAdjacency & adj = m_topology[ic];
		Vector3F dif = adj.getDifferentialCoordinate();
		Matrix33F R = m_targetAnalysis->getR(ic);
		dif = R.transform(dif);
		Vector3F worldP = m_targetAnalysis->restP(ic) + m_targetAnalysis->getT(ic);
		//worldP *= .8f;
		b[0](ic) = dif.x + worldP.x;
		b[1](ic) = dif.y + worldP.y;
		b[2](ic) = dif.z + worldP.z;
	}
	
	b[0] = LT * b[0];
	b[1] = LT * b[1];
	b[2] = LT * b[2];
}

char AccumulateDeformer::solve()
{
/*
	if(m_targetAnalysis->hasNoEffect()) {
		reset();
		return 0;
	}*/
	Eigen::VectorXf b[3];
	prestep(b);
	Eigen::VectorXf x[3];
	x[0] = m_llt.solve(b[0]);
	x[1] = m_llt.solve(b[1]);
	x[2] = m_llt.solve(b[2]);
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_deformedV[i].x = x[0](i);
		m_deformedV[i].y = x[1](i);
		m_deformedV[i].z = x[2](i);
	}

	return 1;
}
