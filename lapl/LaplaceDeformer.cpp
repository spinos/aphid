#include "LaplaceDeformer.h"
#include "VertexAdjacency.h"
#include "MeshLaplacian.h"
#include <Eigen/SVD>
LaplaceDeformer::LaplaceDeformer() {}
LaplaceDeformer::~LaplaceDeformer() {}

void LaplaceDeformer::initialCondition()
{
	int ne = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		ne += adj.getNumNeighbors();
	}
	printf("edge count: %i", ne);
	m_vPi = new Vector3F[ne];
	
	int neighborIdx;
	float neighborWei;
	std::map<int,int> ordered;
	std::map<int,int>::iterator orderIt;
	
	Vector3F * p = m_mesh->getVertices();
	
	ne = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		ordered = adj.getNeighborOrder();
		for (orderIt= ordered.begin(); orderIt!= ordered.end(); ++orderIt) {
			adj.getNeighbor(orderIt->second, neighborIdx, neighborWei);
			m_vPi[ne] = p[i] - p[neighborIdx];
			m_vPi[ne] *= neighborWei;
			ne++;
		}
	}
	
	m_mRi = new Matrix33F[m_numVertices];
}

char LaplaceDeformer::fillM()
{
	std::map<int,int> ordered;
	std::map<int,int>::iterator orderIt;
	int neighborIdx, lastNeighbor;
	float neighborWei;
	LaplaceMatrixType L(m_numVertices + 3, m_numVertices);
	L.setZero();
	L.startFill();
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		lastNeighbor = -1;
		ordered = adj.getNeighborOrder();
		for (orderIt= ordered.begin(); orderIt!= ordered.end(); ++orderIt) {
			adj.getNeighbor(orderIt->second, neighborIdx, neighborWei);
			if(neighborIdx > i && lastNeighbor < i) {
				L.fill(i, i) = -1.0f;
			}
			L.fill(i, neighborIdx) = neighborWei;
			lastNeighbor = neighborIdx; 
		}
		if(lastNeighbor < i)
			L.fill(i, i) = -1.0f;
	}
	L.fill(m_numVertices, 5) = 1.f;
	L.fill(m_numVertices + 1, 47) = 1.f;
	L.fill(m_numVertices + 2, 67) = 1.f;
	L.endFill();
	
	m_LT = L.transpose();
	m_M = m_LT * L;
	//std::cout << "M \n" << m_M << std::endl;
	m_llt = Eigen::SparseLLT<LaplaceMatrixType>(m_M);
    return 1;
}

char LaplaceDeformer::fillDelta()
{
	m_delta[0].resize(m_numVertices + 3);
	m_delta[1].resize(m_numVertices + 3);
	m_delta[2].resize(m_numVertices + 3);
	
	std::map<int,int> ordered;
	std::map<int,int>::iterator orderIt;
	int neighborIdx;
	float neighborWei;
	int allEdgeIdx = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		//m_delta[0](i) = adj.getDeltaCoordX();
		//m_delta[1](i) = adj.getDeltaCoordY();
		//m_delta[2](i) = adj.getDeltaCoordZ();
		m_delta[0](i) = 0.f;
		m_delta[1](i) = 0.f;
		m_delta[2](i) = 0.f;
		ordered = adj.getNeighborOrder();
		for (orderIt= ordered.begin(); orderIt!= ordered.end(); ++orderIt) {
			adj.getNeighbor(orderIt->second, neighborIdx, neighborWei);
			Vector3F pij = m_vPi[allEdgeIdx];
			Matrix33F mixR = m_mRi[i] + m_mRi[neighborIdx];
			pij = mixR.transform(pij);
			pij /= 2.f;
			m_delta[0](i) -= pij.x;
			m_delta[1](i) -= pij.y;
			m_delta[2](i) -= pij.z;
			allEdgeIdx++; 
		}
	}
	
	m_delta[0](m_numVertices) = 16;
	m_delta[1](m_numVertices) = 6;
	m_delta[2](m_numVertices) = 5;
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

void LaplaceDeformer::setMesh(BaseMesh * mesh)
{
	BaseDeformer::setMesh(mesh);
	
	printf("init laplace deformer");
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_mesh);
	m_topology = msh->connectivity();
	fillM();
	initialCondition();
}

void LaplaceDeformer::updateRi()
{
	std::map<int,int> ordered;
	std::map<int,int>::iterator orderIt;
	int neighborIdx;
	float neighborWei;
	Eigen::MatrixXf P, Q, S;
	int degree;
	int allEdgeIdx = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		degree = adj.getNumNeighbors();
		P.resize(3, degree);
		Q.resize(3, degree);
		ordered = adj.getNeighborOrder();
		
		degree = 0;
		for (orderIt= ordered.begin(); orderIt!= ordered.end(); ++orderIt) {
			adj.getNeighbor(orderIt->second, neighborIdx, neighborWei);
			
			P(0, degree) = m_vPi[allEdgeIdx].x;
			P(1, degree) = m_vPi[allEdgeIdx].y;
			P(2, degree) = m_vPi[allEdgeIdx].z;
			
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
		//std::cout<<"Ri \n"<<R<<"\n";
		Matrix33F &pR = m_mRi[i];
		*pR.m(0, 0) = R(0, 0);*pR.m(0, 1) = R(0, 1);*pR.m(0, 2) = R(0, 2);
		*pR.m(1, 0) = R(1, 0);*pR.m(1, 1) = R(1, 1);*pR.m(1, 2) = R(1, 2);
		*pR.m(2, 0) = R(2, 0);*pR.m(2, 1) = R(2, 1);*pR.m(2, 2) = R(2, 2);
		
	}
}

char LaplaceDeformer::solve()
{
	for(int i=0; i < 4; i++) {
		fillDelta();
		
		m_llt.solveInPlace(m_delta[0]);
		m_llt.solveInPlace(m_delta[1]);
		m_llt.solveInPlace(m_delta[2]);
		
		if(i<3)updateRi();
	}
	
	for(int i = 0; i < m_M.rows(); i++) {
		m_deformedV[i].x = m_delta[0](i);
		m_deformedV[i].y = m_delta[1](i);
		m_deformedV[i].z = m_delta[2](i);
	}
	
	
	return 1;
}