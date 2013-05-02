/*
 *  AnchorDeformer.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "AnchorDeformer.h"

#include "VertexAdjacency.h"
#include "AllGeometry.h"

AnchorDeformer::AnchorDeformer() 
{
	m_membrane = 0;
	m_smoothFactor = 1.f;
}

AnchorDeformer::~AnchorDeformer() 
{
	if(m_membrane) delete[] m_membrane;
}

void AnchorDeformer::setMesh(BaseMesh * mesh)
{
	BaseDeformer::setMesh(mesh);
	
	printf("init anchor deformer");
	buildTopology(mesh);
}

void AnchorDeformer::reset()
{
    BaseDeformer::reset();
    m_smoothFactor = 1.f;
}

void AnchorDeformer::setAnchors(AnchorGroup * ag)
{
	m_anchors = ag;
}

void AnchorDeformer::precompute()
{	
	unsigned numConstrains = 0;
	for(Anchor *a = m_anchors->firstAnchor(); m_anchors->hasAnchor(); a = m_anchors->nextAnchor()) {
		numConstrains += a->numPoints();
	}
	
	int neighborIdx;
	m_L.resize(m_numVertices + numConstrains, m_numVertices);
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = getTopology()[i];
		
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
			neighborIdx = neighbor->v->getIndex();
			m_L.insert(i, neighborIdx) = -neighbor->weight;
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
		VertexAdjacency & adj = getTopology()[i];
		Vector3F dif = adj.getDifferentialCoordinate();
		m_delta[0](i) = dif.x;
		m_delta[1](i) = dif.y;
		m_delta[2](i) = dif.z;
	}
	
	if(m_membrane) delete[] m_membrane;
	m_membrane = new Vector3F[m_numVertices];
	
	Eigen::VectorXd b[3];
	prestep(b, 1);
	Eigen::VectorXd x[3];
	x[0] = m_llt.solve(b[0]);
	x[1] = m_llt.solve(b[1]);
	x[2] = m_llt.solve(b[2]);
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_membrane[i].x = x[0](i);
		m_membrane[i].y = x[1](i);
		m_membrane[i].z = x[2](i);
	}
}

void AnchorDeformer::prestep(Eigen::VectorXd b[], char isMembrane)
{	
	b[0] = m_delta[0];
	b[1] = m_delta[1];
	b[2] = m_delta[2];
	
	if(isMembrane) {
		b[0].setZero();
		b[1].setZero();
		b[2].setZero();
	}
	else {
		for(int i = 0; i < (int)m_numVertices; i++) {
			Matrix33F R = svdRotation(i);
			Vector3F dif(b[0](i), b[1](i), b[2](i));
			dif = R.transform(dif);
			b[0](i) = dif.x * m_smoothFactor;
			b[1](i) = dif.y * m_smoothFactor;
			b[2](i) = dif.z * m_smoothFactor;
		}
	}
	
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

char AnchorDeformer::solve()
{
	Eigen::VectorXd b[3];
	prestep(b, 1);
	Eigen::VectorXd x[3];
	x[0] = m_llt.solve(b[0]);
	x[1] = m_llt.solve(b[1]);
	x[2] = m_llt.solve(b[2]);
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_deformedV[i].x = x[0](i);
		m_deformedV[i].y = x[1](i);
		m_deformedV[i].z = x[2](i);
	}
	
	prestep(b);
	x[0] = m_llt.solve(b[0]);
	x[1] = m_llt.solve(b[1]);
	x[2] = m_llt.solve(b[2]);
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_deformedV[i].x = x[0](i);
		m_deformedV[i].y = x[1](i);
		m_deformedV[i].z = x[2](i);
	}
	/*
	for(Anchor *a = m_anchors->firstAnchor(); m_anchors->hasAnchor(); a = m_anchors->nextAnchor()) {
		unsigned iap;
		for(Anchor::AnchorPoint *ap = a->firstPoint(iap); a->hasPoint(); ap = a->nextPoint(iap)) {
			Vector3F worldP = ap->worldP;
			m_deformedV[iap] = worldP;
		}	
	}
*/
	return 1;
}

Matrix33F AnchorDeformer::svdRotation(unsigned iv)
{	
	Vector3F v, vt, c, ct, dx, dy;
	int neighborIdx, numNeighbors;
	Eigen::MatrixXf D(3, 3);
	D.setIdentity();
		
	VertexAdjacency & adj = getTopology()[iv];
	numNeighbors = adj.getNumNeighbors();
	
	c.setZero();
	ct.setZero();
	
	VertexAdjacency::VertexNeighbor *neighbor;
	for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
		neighborIdx = neighbor->v->getIndex();
		v = m_membrane[neighborIdx];
		vt = m_deformedV[neighborIdx];
		
		c += v * neighbor->weight;
		ct += vt * neighbor->weight;
	}

	Eigen::MatrixXf X(3, numNeighbors);
	Eigen::MatrixXf Y(3, numNeighbors);
	Eigen::MatrixXf W(numNeighbors, numNeighbors);
	W.setZero();
	int k = 0;
	for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
		neighborIdx = neighbor->v->getIndex();
		v = m_membrane[neighborIdx];
		vt = m_deformedV[neighborIdx];
		
		dx = v - c;
		dy = vt - ct;
		
		X(0, k) = dx.x;
		X(1, k) = dx.y;
		X(2, k) = dx.z;
		
		Y(0, k) = dy.x;
		Y(1, k) = dy.y;
		Y(2, k) = dy.z;
		
		W(k, k) = neighbor->weight;
		k++;
	}
	
	Eigen::MatrixXf S = X * W * Y.transpose();
	
	Eigen::JacobiSVD<Eigen::MatrixXf > solver(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
	
	float d = (solver.matrixV() * solver.matrixU().transpose()).determinant();
	D(2,2) = d;
	
	Eigen::MatrixXf R = solver.matrixV() * D * solver.matrixU().transpose();
	
	Matrix33F Ri;
	*Ri.m(0, 0) = R(0, 0);
	*Ri.m(0, 1) = R(1, 0);
	*Ri.m(0, 2) = R(2, 0);
	*Ri.m(1, 0) = R(0, 1);
	*Ri.m(1, 1) = R(1, 1);
	*Ri.m(1, 2) = R(2, 1);
	*Ri.m(2, 0) = R(0, 2);
	*Ri.m(2, 1) = R(1, 2);
	*Ri.m(2, 2) = R(2, 2);
	return Ri;
}

float AnchorDeformer::getSmoothFactor() const
{
    return m_smoothFactor;
}

void AnchorDeformer::setSmoothFactor(float factor)
{
    m_smoothFactor = factor;
}
