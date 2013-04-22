/*
 *  MembraneDeformer.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MembraneDeformer.h"

#include "VertexAdjacency.h"
#include "AllGeometry.h"

MembraneDeformer::MembraneDeformer() 
{
}

MembraneDeformer::~MembraneDeformer() {}

void MembraneDeformer::setMesh(BaseMesh * mesh)
{
	BaseDeformer::setMesh(mesh);
	
	printf("init membrane deformer");
	buildTopology();
}

void MembraneDeformer::setAnchors(AnchorGroup * ag)
{
	m_anchors = ag;
}

void MembraneDeformer::precompute()
{	
	unsigned numConstrains = 0;
	for(Anchor *a = m_anchors->firstAnchor(); m_anchors->hasAnchor(); a = m_anchors->nextAnchor()) {
		numConstrains += a->numPoints();
	}
	
	int neighborIdx;
	m_L.resize(m_numVertices + numConstrains, m_numVertices);
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
			neighborIdx = neighbor->v->getIndex();
			m_L.insert(i, neighborIdx) = -1.f / adj.getNumNeighbors();
		}
		m_L.insert(i, i) = 1.f;
	}
	
	unsigned irow = m_numVertices;
	for(Anchor *a = m_anchors->firstAnchor(); m_anchors->hasAnchor(); a = m_anchors->nextAnchor()) {
		unsigned iap;
		for(Anchor::AnchorPoint *ap = a->firstPoint(iap); a->hasPoint(); ap = a->nextPoint(iap)) {
			m_L.coeffRef(irow, iap) = 1.f;
			irow++;
		}	
	}
	
	LaplaceMatrixType L = m_L;
	
	m_LT = L.transpose();
	
	LaplaceMatrixType M = m_LT * L;
	m_llt.compute(M);
	
	m_delta[0].resize(m_numVertices + numConstrains);
	m_delta[1].resize(m_numVertices + numConstrains);
	m_delta[2].resize(m_numVertices + numConstrains);
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		Vector3F dif = adj.getDifferentialCoordinate();
		m_delta[0](i) = dif.x;
		m_delta[1](i) = dif.y;
		m_delta[2](i) = dif.z;
	}
}

void MembraneDeformer::prestep(Eigen::VectorXd b[])
{	
	b[0] = m_delta[0];
	b[1] = m_delta[1];
	b[2] = m_delta[2];
	
	b[0].setZero();
	b[1].setZero();
	b[2].setZero();
	
	unsigned irow = m_numVertices;
	for(Anchor *a = m_anchors->firstAnchor(); m_anchors->hasAnchor(); a = m_anchors->nextAnchor()) {
		unsigned iap;
		for(Anchor::AnchorPoint *ap = a->firstPoint(iap); a->hasPoint(); ap = a->nextPoint(iap)) {
			Vector3F worldP = ap->worldP;
			b[0](irow) = worldP.x;
			b[1](irow) = worldP.y;
			b[2](irow) = worldP.z;
			irow++;
		}	
	}
	
	b[0] = m_LT * b[0];
	b[1] = m_LT * b[1];
	b[2] = m_LT * b[2];
}

char MembraneDeformer::solve()
{
	Eigen::VectorXd b[3];
	prestep(b);
	Eigen::VectorXd x[3];
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

char MembraneDeformer::buildTopology()
{
	const unsigned nv = m_mesh->getNumVertices();
	
	m_topology = new VertexAdjacency[nv];
	
	for(unsigned i = 0; i < nv; i++) {
		VertexAdjacency & v = m_topology[i];
		v.setIndex(i);
		v.m_v = &(m_mesh->getVertices()[i]);
	}
	
	const unsigned nf = m_mesh->getNumFaces();
	unsigned a, b, c;
	
	for(unsigned i = 0; i < nf; i++) {
		a = m_mesh->getIndices()[i * 3];
		b = m_mesh->getIndices()[i * 3 + 1];
		c = m_mesh->getIndices()[i * 3 + 2];
		Facet * f = new Facet(&m_topology[a], &m_topology[b], &m_topology[c]);
		f->setIndex(i);
		for(unsigned j = 0; j < 3; j++) {
			Edge * e = f->edge(j);
			m_topology[e->v0()->getIndex()].addEdge(e);
			m_topology[e->v1()->getIndex()].addEdge(e);
		}
	}
	
	for(unsigned i = 0; i < nv; i++) {
		m_topology[i].findNeighbors();
		m_topology[i].computeWeights();
		m_topology[i].computeDifferentialCoordinate();
	}
	return 1;
}
