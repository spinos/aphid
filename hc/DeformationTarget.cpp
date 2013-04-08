/*
 *  DeformationTarget.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "DeformationTarget.h"
#include "VertexAdjacency.h"
#include "MeshLaplacian.h"
#include "HarmonicCoord.h"
#include "Matrix44F.h"

DeformationTarget::DeformationTarget() {}
DeformationTarget::~DeformationTarget() 
{
	delete[] m_Ri;
	delete[] m_scale;
}
	
void DeformationTarget::setMeshes(BaseMesh * a, BaseMesh * b)
{
	m_restMesh = a;
	m_effectMesh = b;
	
	computeR();
	edgeScale();
	computeTRange();
}

void DeformationTarget::setWeightMap(HarmonicCoord * hc, unsigned valueId)
{
	m_weightMap = hc;
	m_valueId = valueId;
}

BaseMesh * DeformationTarget::getMeshA() const
{
	return m_restMesh;
}

BaseMesh * DeformationTarget::getMeshB() const
{
	return m_effectMesh;
}

void DeformationTarget::computeR()
{
	svdRotation();
}

void DeformationTarget::svdRotation()
{
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_restMesh);
	VertexAdjacency *topology = msh->connectivity();
	
	unsigned numVertices = m_restMesh->getNumVertices();
	const Vector3F * vs = m_restMesh->getVertices();
	const Vector3F * vts = m_effectMesh->getVertices();
	
	m_Ri = new Matrix33F[numVertices];
	
	Vector3F v, vt, c, ct, dx, dy;
	int neighborIdx, numNeighbors;
	Eigen::MatrixXf D(3, 3);
	D.setIdentity();
		
	for(int i = 0; i < (int)numVertices; i++) {
		VertexAdjacency & adj = topology[i];
		numNeighbors = adj.getNumNeighbors();
		
		c.setZero();
		ct.setZero();
		
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
			neighborIdx = neighbor->v->getIndex();
			v = vs[neighborIdx];
			vt = vts[neighborIdx];
			
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
			v = vs[neighborIdx];
			vt = vts[neighborIdx];
			
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
		
		Matrix33F &Ri = m_Ri[i];
		*Ri.m(0, 0) = R(0, 0);
		*Ri.m(0, 1) = R(1, 0);
		*Ri.m(0, 2) = R(2, 0);
		*Ri.m(1, 0) = R(0, 1);
		*Ri.m(1, 1) = R(1, 1);
		*Ri.m(1, 2) = R(2, 1);
		*Ri.m(2, 0) = R(0, 2);
		*Ri.m(2, 1) = R(1, 2);
		*Ri.m(2, 2) = R(2, 2);
	}
	
}

void DeformationTarget::edgeScale()
{
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_restMesh);
	VertexAdjacency *topology = msh->connectivity();
	
	unsigned numVertices = m_restMesh->getNumVertices();
	const Vector3F * vs = m_restMesh->getVertices();
	const Vector3F * vts = m_effectMesh->getVertices();
	
	m_scale = new float[numVertices];
	
	Vector3F v, vt, c, ct, dx, dy;
	int neighborIdx;
		
	for(int i = 0; i < (int)numVertices; i++) {
		VertexAdjacency & adj = topology[i];
		
		c.setZero();
		ct.setZero();
		
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
			neighborIdx = neighbor->v->getIndex();
			v = vs[neighborIdx];
			vt = vts[neighborIdx];
			
			c += v * neighbor->weight;
			ct += vt * neighbor->weight;
		}
		
		m_scale[i] = 0.f;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
			neighborIdx = neighbor->v->getIndex();
			v = vs[neighborIdx];
			vt = vts[neighborIdx];
			
			dx = v - c;
			dy = vt - ct;
			
			m_scale[i] += (dy.length() / dx.length()) * neighbor->weight;
		}
	}
}

void DeformationTarget::computeTRange()
{
	m_minDisplacement = 10e28;
	m_maxDisplacement = -10e28;
	
	unsigned numVertices = m_restMesh->getNumVertices();
	const Vector3F * vs = m_restMesh->getVertices();
	const Vector3F * vts = m_effectMesh->getVertices();
	
	for(int i = 0; i < (int)numVertices; i++) {
		Vector3F di = vts[i] - vs[i];
		const float d = di.length();
		if(d < 0.03f) continue;
		
		if(d < m_minDisplacement) m_minDisplacement = d;
		if(d > m_maxDisplacement) m_maxDisplacement = d;
	}
}

unsigned DeformationTarget::numVertices() const
{
	return m_restMesh->getNumVertices();
}

Vector3F DeformationTarget::restP(unsigned idx) const
{
	return m_restMesh->getVertices()[idx];
}

Vector3F DeformationTarget::differential(unsigned idx) const
{
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_restMesh);
	return msh->connectivity()[idx].getDifferentialCoordinate();
}

Vector3F DeformationTarget::transformedDifferential(unsigned idx) const
{
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_restMesh);
	Vector3F d = msh->connectivity()[idx].getDifferentialCoordinate();
	d = m_Ri[idx].transform(d);
	return d;
}

Matrix33F DeformationTarget::getR(unsigned idx) const
{
	return m_Ri[idx];
}

Vector3F DeformationTarget::getT(unsigned idx) const
{
	const Vector3F * vs = m_restMesh->getVertices();
	const Vector3F * vts = m_effectMesh->getVertices();
	return (vts[idx] - vs[idx]) * m_weightMap->getValue(m_valueId, idx);
}

float DeformationTarget::getS(unsigned idx) const
{
	return m_scale[idx];
}

float DeformationTarget::getConstrainWeight(unsigned idx) const
{
	return m_weightMap->getValue(m_valueId, idx);
}

float DeformationTarget::minDisplacement() const
{
	return m_minDisplacement;
}

void DeformationTarget::genNonZeroIndices(std::map<unsigned, char > & dst) const
{
	unsigned numVertices = m_restMesh->getNumVertices();
	for(unsigned i = 0; i < numVertices; i++) {
		if(m_weightMap->getValue(m_valueId, i) > 10e-3)
			dst[i] = 1;
	}
}

