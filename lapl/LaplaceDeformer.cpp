#include "LaplaceDeformer.h"
#include "VertexAdjacency.h"
LaplaceDeformer::LaplaceDeformer() {}
LaplaceDeformer::~LaplaceDeformer() {}

char LaplaceDeformer::fillM(const unsigned & numVertices, VertexAdjacency * adjacency)
{
	std::map<int,int> ordered;
	std::map<int,int>::iterator orderIt;
	int neighborIdx, lastNeighbor = -1;
	float neighborWei;
	LaplaceMatrixType L(numVertices + 3, numVertices);
	L.setZero();
	L.startFill();
	for(int i = 0; i < (int)numVertices; i++) {
		VertexAdjacency & adj = adjacency[i];

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
	L.fill(numVertices, 0) = 1.f;
	L.fill(numVertices + 1, 47) = 1.f;
	L.fill(numVertices + 2, 67) = 1.f;
	L.endFill();
	
	m_LT = L.transpose();
	m_M = m_LT * L;
	//std::cout << "M \n" << m_M << std::endl;
	m_llt = Eigen::SparseLLT<LaplaceMatrixType>(m_M);
    return 1;
}

char LaplaceDeformer::fillDelta(const unsigned & numVertices, VertexAdjacency * adjacency)
{
	m_delta[0].resize(numVertices + 3);
	m_delta[1].resize(numVertices + 3);
	m_delta[2].resize(numVertices + 3);
	
	for(int i = 0; i < (int)numVertices; i++) {
		VertexAdjacency & adj = adjacency[i];
		m_delta[0](i) = adj.getDeltaCoordX();
		m_delta[1](i) = adj.getDeltaCoordY();
		m_delta[2](i) = adj.getDeltaCoordZ();
	}
	
	m_delta[0](numVertices) = 11;
	m_delta[1](numVertices) = 2;
	m_delta[2](numVertices) = 5;
	m_delta[0](numVertices + 1) = 5;
	m_delta[1](numVertices + 1) = 25;
	m_delta[2](numVertices + 1) = 18;
	m_delta[0](numVertices + 2) = 23;
	m_delta[1](numVertices + 2) = 22;
	m_delta[2](numVertices + 2) = 20;
	
	m_delta[0] = m_LT * m_delta[0];
	m_delta[1] = m_LT * m_delta[1];
	m_delta[2] = m_LT * m_delta[2];
	return 1;
}

char LaplaceDeformer::solve()
{
	m_llt.solveInPlace(m_delta[0]);
	m_llt.solveInPlace(m_delta[1]);
	m_llt.solveInPlace(m_delta[2]);
	
	for(int i = 0; i < m_M.rows(); i++) {
		m_deformedV[i].x = m_delta[0](i);
		m_deformedV[i].y = m_delta[1](i);
		m_deformedV[i].z = m_delta[2](i);
	}
	return 1;
}