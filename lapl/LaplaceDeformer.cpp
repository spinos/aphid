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
	return 8;
}

void LaplaceDeformer::initialCondition()
{
	int ne = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		ne += adj.getNumNeighbors();
	}
	printf("edge count: %i\n", ne);
	m_ViVjWj = new Vector3F[ne];
	
	Vector3F * p = m_mesh->getVertices();
	m_mRi = new Eigen::Matrix3f[m_numVertices];
	ne = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
            m_ViVjWj[ne] = p[i] - * neighbor->v;
			ne++;
		}
		m_mRi[i].setIdentity();
	}
	printf("edge count: %i\n", ne);
	
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
				L.fill(i, i) = 1.0f;
			}
			L.fill(i, neighborIdx) = -neighbor->weight;
			lastNeighbor = neighborIdx; 
		}
		if(lastNeighbor < i)
			L.fill(i, i) = 1.0f;
	}
	L.fill(m_numVertices, 0) = 1.f;
	L.fill(m_numVertices + 1, 8) = 1.f;
	L.fill(m_numVertices + 2, 656) = 1.f;
	L.fill(m_numVertices + 3, 648) = 1.f;
	L.fill(m_numVertices + 4, 288) = 0.5f;
	L.fill(m_numVertices + 5, 296) = 0.5f;
	L.fill(m_numVertices + 6, 368) = 0.5f;
	L.fill(m_numVertices + 7, 360) = 0.5f;

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
		    //Vector3F pij = m_ViVjWj[allEdgeIdx];
			Vector3F pij = m_deformedV[i] - m_deformedV[neighbor->v->getIndex()];
			Eigen::Vector3f matpij;
			matpij(0) = pij.x;
			matpij(1) = pij.y;
			matpij(2) = pij.z;
			Eigen::Matrix3f mixR = m_mRi[i] + m_mRi[neighbor->v->getIndex()];
			matpij = mixR * matpij;
			m_delta[0](i) += matpij(0) * neighbor->weight / 2.f;
			m_delta[1](i) += matpij(1) * neighbor->weight / 2.f;
			m_delta[2](i) += matpij(2) * neighbor->weight / 2.f;
			allEdgeIdx++;
		}
	}
	
	m_delta[0](m_numVertices) = -4 + 10;
	m_delta[1](m_numVertices) = 0;
	m_delta[2](m_numVertices) = 4;
	
	m_delta[0](m_numVertices + 1) = 4 + 10;
	m_delta[1](m_numVertices + 1) = 0;
	m_delta[2](m_numVertices + 1) = 4;
	
	m_delta[0](m_numVertices + 2) = 4+ 10;
	m_delta[1](m_numVertices + 2) = 0;
	m_delta[2](m_numVertices + 2) = -4;
	
	m_delta[0](m_numVertices + 3) = -4+ 10;
	m_delta[1](m_numVertices + 3) = 0;
	m_delta[2](m_numVertices + 3) = -4;
	
	Vector3F side(1, 0, 0);
	Vector3F up(0, 1, 0);
	Vector3F front(0,0,1);
	Matrix33F rot;
	rot.fill(side, up, front);
	
	Vector3F trans(28, 19, 0);
	Vector3F a = rot.transform(Vector3F(-4, 0, 4)) + trans;
	m_delta[0](m_numVertices + 4) = a.x * 0.5f;
	m_delta[1](m_numVertices + 4) = a.y * 0.5f;
	m_delta[2](m_numVertices + 4) = a.z * 0.5f;
	
	Vector3F b = rot.transform(Vector3F(4, 0, 4)) + trans;
	m_delta[0](m_numVertices + 5) = b.x * 0.5f;
	m_delta[1](m_numVertices + 5) = b.y * 0.5f;
	m_delta[2](m_numVertices + 5) = b.z * 0.5f;
	
	Vector3F c = rot.transform(Vector3F(4, 0, -4)) + trans;
	m_delta[0](m_numVertices + 6) = c.x * 0.5f;
	m_delta[1](m_numVertices + 6) = c.y * 0.5f;
	m_delta[2](m_numVertices + 6) = c.z * 0.5f;
	
	Vector3F d = rot.transform(Vector3F(-4, 0, -4)) + trans;
	
	m_delta[0](m_numVertices + 7) = d.x * 0.5f;
	m_delta[1](m_numVertices + 7) = d.y * 0.5f;
	m_delta[2](m_numVertices + 7) = d.z * 0.5f;

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
/*
			P(0, degree) = m_ViVjWj[allEdgeIdx].x * neighbor->weight;
			P(1, degree) = m_ViVjWj[allEdgeIdx].y * neighbor->weight;
			P(2, degree) = m_ViVjWj[allEdgeIdx].z * neighbor->weight;
*/			
			Vector3F vp = m_deformedV[i] - m_deformedV[neighborIdx];
			
			P(0, degree) = vp.x * neighbor->weight;
			P(1, degree) = vp.y * neighbor->weight;
			P(2, degree) = vp.z * neighbor->weight;

			Q(0, degree) = m_delta[0](i) - m_delta[0](neighborIdx);
			Q(1, degree) = m_delta[1](i) - m_delta[1](neighborIdx);
			Q(2, degree) = m_delta[2](i) - m_delta[2](neighborIdx);
			allEdgeIdx++;
			degree++;
		}
		S = P * Q.transpose();
		
		Eigen::SVD<Eigen::MatrixXf > svdSolver(S);
		
		Eigen::MatrixXf R = svdSolver.matrixV() * svdSolver.matrixU().transpose();
		float d = R.determinant();
		
		Eigen::MatrixXf dd;
		dd.resize(3,3);
		dd(0,0) = 1; dd(0,1) = 0;dd(0,2) = 0;
		dd(1,1) = 0; dd(1,1) = 1;dd(1,2) = 0;
		dd(2,0) = 0; dd(2,1) = 0;dd(2,2) = d;
		
		m_mRi[i] = svdSolver.matrixV() * dd * svdSolver.matrixU().transpose();
		
		//std::cout<<" "<<m_mRi[i]<<std::endl;
		//std::cout<<" "<<d<<std::endl;
		
	}
}

char LaplaceDeformer::solve()
{
	int nstep = 4;
	for(int i=0; i < nstep; i++) {
		fillDelta();
		
		m_delta[0] = m_LT * m_delta[0];
		m_delta[1] = m_LT * m_delta[1];
		m_delta[2] = m_LT * m_delta[2];
		
		m_llt.solveInPlace(m_delta[0]);
		m_llt.solveInPlace(m_delta[1]);
		m_llt.solveInPlace(m_delta[2]);
		
		if(i != nstep-1) updateRi();
	}
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_deformedV[i].x = m_delta[0](i);
		m_deformedV[i].y = m_delta[1](i);
		m_deformedV[i].z = m_delta[2](i);
	}

	return 1;
}