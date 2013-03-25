/*
 *  DeformationAnalysis.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "DeformationAnalysis.h"
#include "LaplaceDeformer.h"
#include <Eigen/SVD>
#include "VertexAdjacency.h"
#include "MeshLaplacian.h"
#include "Matrix44F.h"

DeformationAnalysis::DeformationAnalysis() {}
DeformationAnalysis::~DeformationAnalysis() {}
	
void DeformationAnalysis::setMeshes(BaseMesh * a, BaseMesh * b)
{
	m_restMesh = a;
	m_effectMesh = b;
	
	computeR();
}

BaseMesh * DeformationAnalysis::getMeshA() const
{
	return m_restMesh;
}

BaseMesh * DeformationAnalysis::getMeshB() const
{
	return m_effectMesh;
}

void DeformationAnalysis::computeR()
{
	svdRotation();
}

void DeformationAnalysis::svdRotation()
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
		
		numNeighbors = adj.getNumNeighbors();
	
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
		
		Eigen::SVD<Eigen::MatrixXf > solver(S);
		
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

void DeformationAnalysis::shtRotation()
{
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_restMesh);
	VertexAdjacency *topology = msh->connectivity();
	
	unsigned numVertices = m_restMesh->getNumVertices();
	const Vector3F * vs = m_restMesh->getVertices();
	const Vector3F * vts = m_effectMesh->getVertices();
	
	m_Ri = new Matrix33F[numVertices];
	
	Vector3F v, vt;
	int neighborIdx;
	for(int i = 0; i < (int)numVertices; i++) {
		VertexAdjacency & adj = topology[i];
	
		Eigen::MatrixXf A(adj.getNumNeighbors() * 3, 7);
		A.setZero();
		Eigen::VectorXf b;
		b.resize(adj.getNumNeighbors() * 3);
		
		VertexAdjacency::VertexNeighbor *neighbor;
		int k = 0;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
			neighborIdx = neighbor->v->getIndex();
			v = vs[neighborIdx];
			A(k * 3    , 0) =  v.x;
			A(k * 3    , 2) =  v.z;
			A(k * 3    , 3) = -v.y;
			A(k * 3    , 4) = 1.0;
			A(k * 3 + 1, 0) =  v.y;
			A(k * 3 + 1, 1) = -v.z;
			A(k * 3 + 1, 3) =  v.x;
			A(k * 3 + 1, 5) = 1.0;
			A(k * 3 + 2, 0) =  v.z;
			A(k * 3 + 2, 1) =  v.y;
			A(k * 3 + 2, 2) = -v.x;
			A(k * 3 + 2, 6) = 1.0;
			
			vt = vts[neighborIdx];
			b(k * 3) = vt.x;
			b(k * 3 + 1) = vt.y;
			b(k * 3 + 2) = vt.z;
			k++;
		}
		//std::cout<<" A"<<A<<"]\n";
		Eigen::MatrixXf AT = A.transpose();
	
		A = AT * A;
		
		b = AT * b;
		
		Eigen::LU<Eigen::MatrixXf> lu(A);
		
		Eigen::VectorXf x(7);
		
		lu.solve(b, &x);
		
		std::cout<<" x "<<x<<"]\n";
		
		Matrix33F &R = m_Ri[i];
		*R.m(0, 0) =  x(0);
		*R.m(0, 1) =  x(3);
		*R.m(0, 2) = -x(2);
		*R.m(1, 0) = -x(3);
		*R.m(1, 1) =  x(0);
		*R.m(1, 2) =  x(1);
		*R.m(2, 0) =  x(2);
		*R.m(2, 1) = -x(1);
		*R.m(2, 2) =  x(0);
	}
}

unsigned DeformationAnalysis::numVertices() const
{
	return m_restMesh->getNumVertices();
}

Vector3F DeformationAnalysis::restP(unsigned idx) const
{
	return m_restMesh->getVertices()[idx];
}

Vector3F DeformationAnalysis::differential(unsigned idx) const
{
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_restMesh);
	return msh->connectivity()[idx].getDifferentialCoordinate();
}

Vector3F DeformationAnalysis::transformedDifferential(unsigned idx) const
{
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_restMesh);
	Vector3F d = msh->connectivity()[idx].getDifferentialCoordinate();
	d = m_Ri[idx].transform(d);
	return d;
}


