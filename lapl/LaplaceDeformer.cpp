#include "LaplaceDeformer.h"
#include "VertexAdjacency.h"
#include "MeshLaplacian.h"
#include <Eigen/SVD>
LaplaceDeformer::LaplaceDeformer() {}
LaplaceDeformer::~LaplaceDeformer() {}

void LaplaceDeformer::setMesh(BaseMesh * mesh)
{
	BaseDeformer::setMesh(mesh);
	
	printf("init laplace deformer");
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_mesh);
	m_topology = msh->connectivity();
	fillL();
	initialCondition();
}

unsigned LaplaceDeformer::getNumAnchors() const
{
	return 3;
}

void LaplaceDeformer::initialCondition()
{
	int ne = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		ne += adj.getNumNeighbors();
	}
	printf("edge count: %i", ne);
	m_ViVjWj = new Vector3F[ne];
	
	Vector3F * p = m_mesh->getVertices();
	
	ne = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
            m_ViVjWj[ne] = p[i] - * neighbor->v;
			m_ViVjWj[ne] *= neighbor->weight;
			ne++;
		}
	}
	
	m_mRi = new Matrix33F[m_numVertices];
}

char LaplaceDeformer::fillL()
{
	int neighborIdx, lastNeighbor;
	LaplaceMatrixType L(m_numVertices + getNumAnchors(), m_numVertices);
	L.setZero();
	L.startFill();
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		lastNeighbor = -1;
		
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
		    neighborIdx = neighbor->v->getIndex();
			if(neighborIdx > i && lastNeighbor < i) {
				L.fill(i, i) = -1.0f;
			}
			L.fill(i, neighborIdx) = neighbor->weight;
			lastNeighbor = neighborIdx; 
		}
		if(lastNeighbor < i)
			L.fill(i, i) = -1.0f;
	}
	L.fill(m_numVertices, 5) = 1.f;
	L.fill(m_numVertices + 1, 47) = 1.f;
	L.fill(m_numVertices + 2, 65) = 1.f;
	L.endFill();
	
	m_LT = L.transpose();
	LaplaceMatrixType m_M = m_LT * L;
	//std::cout << "M \n" << m_M << std::endl;
	m_llt = Eigen::SparseLLT<LaplaceMatrixType>(m_M);
    return 1;
}

char LaplaceDeformer::fillDelta()
{
	const unsigned numAchors = getNumAnchors();
	m_delta[0].resize(m_numVertices + numAchors);
	m_delta[1].resize(m_numVertices + numAchors);
	m_delta[2].resize(m_numVertices + numAchors);
	
	int allEdgeIdx = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];

		m_delta[0](i) = 0.f;
		m_delta[1](i) = 0.f;
		m_delta[2](i) = 0.f;
		
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
		    Vector3F pij = m_ViVjWj[allEdgeIdx];
			Matrix33F mixR = m_mRi[i] + m_mRi[neighbor->v->getIndex()];
			pij = mixR.transform(pij);
			pij /= 2.f;
			m_delta[0](i) -= pij.x;
			m_delta[1](i) -= pij.y;
			m_delta[2](i) -= pij.z;
			allEdgeIdx++;
		}
	}
	
	m_delta[0](m_numVertices) = 12;
	m_delta[1](m_numVertices) = 3;
	m_delta[2](m_numVertices) = 7;
	m_delta[0](m_numVertices + 1) = 1;
	m_delta[1](m_numVertices + 1) = 25;
	m_delta[2](m_numVertices + 1) = 7;
	m_delta[0](m_numVertices + 2) = 17;
	m_delta[1](m_numVertices + 2) = 13;
	m_delta[2](m_numVertices + 2) = 22;
	
	m_delta[0] = m_LT * m_delta[0];
	m_delta[1] = m_LT * m_delta[1];
	m_delta[2] = m_LT * m_delta[2];
	return 1;
}

void LaplaceDeformer::updateRi()
{
	int neighborIdx;
	Eigen::MatrixXf P, Q, S;
	int degree;
	int allEdgeIdx = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		degree = adj.getNumNeighbors();
		P.resize(3, degree);
		Q.resize(3, degree);

		degree = 0;
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {

			neighborIdx = neighbor->v->getIndex();
			
			P(0, degree) = m_ViVjWj[allEdgeIdx].x;
			P(1, degree) = m_ViVjWj[allEdgeIdx].y;
			P(2, degree) = m_ViVjWj[allEdgeIdx].z;
			
			Q(0, degree) = m_delta[0](i) - m_delta[0](neighborIdx);
			Q(1, degree) = m_delta[1](i) - m_delta[1](neighborIdx);
			Q(2, degree) = m_delta[2](i) - m_delta[2](neighborIdx);
			allEdgeIdx++;
			degree++;
		}
		S = P * Q.transpose();
		
		Eigen::SVD<Eigen::MatrixXf > svdSolver(S);
		Eigen::MatrixXf R = svdSolver.matrixU() * svdSolver.matrixV().transpose();
		
		float d = R.determinant();
		
		Eigen::MatrixXf dd;
		dd.resize(3,3);
		dd.setZero();
		dd(0,0) = d;
		dd(1,1) = d;
		dd(2,2) = d;
		
		R = svdSolver.matrixU() * dd * svdSolver.matrixV().transpose();

		Matrix33F &pR = m_mRi[i];
		*pR.m(0, 0) = R(0, 0);*pR.m(0, 1) = R(0, 1);*pR.m(0, 2) = R(0, 2);
		*pR.m(1, 0) = R(1, 0);*pR.m(1, 1) = R(1, 1);*pR.m(1, 2) = R(1, 2);
		*pR.m(2, 0) = R(2, 0);*pR.m(2, 1) = R(2, 1);*pR.m(2, 2) = R(2, 2);
		
	}
}

char LaplaceDeformer::solve()
{
	for(int i=0; i < 5; i++) {
		fillDelta();
		
		m_llt.solveInPlace(m_delta[0]);
		m_llt.solveInPlace(m_delta[1]);
		m_llt.solveInPlace(m_delta[2]);
		
		if(i<4) updateRi();
	}
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_deformedV[i].x = m_delta[0](i);
		m_deformedV[i].y = m_delta[1](i);
		m_deformedV[i].z = m_delta[2](i);
	}

	return 1;
}