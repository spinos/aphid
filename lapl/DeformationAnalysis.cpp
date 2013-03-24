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
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_restMesh);
	VertexAdjacency *topology = msh->connectivity();
	
	unsigned numVertices = m_restMesh->getNumVertices();
	const Vector3F * vs = m_restMesh->getVertices();
	const Vector3F * vts = m_effectMesh->getVertices();
	
	m_sht = new float[numVertices * 7];
	
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
		
		for(int j = 0; j < 7; j++) 
			m_sht[i * 7 + j] = x(j);
	}
}

