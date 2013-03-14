#include "RotateInvariantDeformer.h"
#include "VertexAdjacency.h"
#include "MeshLaplacian.h"
RotateInvariantDeformer::RotateInvariantDeformer() {}
RotateInvariantDeformer::~RotateInvariantDeformer() {}

char RotateInvariantDeformer::fillMRot(VertexAdjacency * adjacency, LaplaceMatrixType &ATA, LaplaceMatrixType &AT)
{
    MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_mesh);
    const unsigned numEdges = countEdges(m_numVertices, adjacency);
    
	std::map<int,int> ordered;
	std::map<int,int>::iterator orderIt;
	int neighborIdx;
	float neighborWei;
	unsigned i3, j3;
	
	const unsigned numAnchors = 2;
	const unsigned anchorIds[2] = {0, 67};
	
	LaplaceMatrixType Rs(numEdges * 3 * 2 + numAnchors, m_numVertices * 3);
	Rs.setZero();
	Rs.startFill();
	
	unsigned rowIdx = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = adjacency[i];
		Matrix33F Fi = adj.getTangentFrame();
		
		i3 = i * 3;

		ordered = adj.getNeighborOrder();
		for (orderIt= ordered.begin(); orderIt!= ordered.end(); ++orderIt) {
			adj.getNeighbor(orderIt->second, neighborIdx, neighborWei);
			
			Matrix33F Fj = msh->getTangentFrame(neighborIdx);
			Fj.transpose();
			Matrix33F Rij = Fj.multiply(Fi);
			
			j3 = neighborIdx * 3;
			
			for ( int k = 0; k < 3; ++k ) {  
				Rs.fill( rowIdx, i3+0 ) = neighborWei * Rij(k, 0);
				Rs.fill( rowIdx, i3+1 ) = neighborWei * Rij(k, 1);
				Rs.fill( rowIdx, i3+2 ) = neighborWei * Rij(k, 2);
				rowIdx++;
				Rs.fill( rowIdx, j3+k ) = -1.0f;
				rowIdx++;
			} 
		}
	}
	
	for(int i=0; i < (int)numAnchors; i++) {
		unsigned anchorIdx = anchorIds[i];
		Rs.fill(rowIdx + i, anchorIdx * 3) = 1;
		Rs.fill(rowIdx + i, anchorIdx * 3 + 1) = 1;
		Rs.fill(rowIdx + i, anchorIdx * 3 + 2) = 1;
	}
	
	Rs.endFill();
	
	AT = Rs.transpose();
	ATA = AT * Rs;
	//std::cout << "M \n" << m_M << std::endl;
	//m_llt = Eigen::SparseLLT<LaplaceMatrixType>(m_M);
    return 1;
}

char RotateInvariantDeformer::solveRotation(MeshLaplacian *msh)
{
	LaplaceMatrixType ATA, AT;
	fillMRot(msh->connectivity(), ATA, AT);
	
	const unsigned numAnchors = 2;
	const unsigned anchorIds[2] = {0, 67};
	
	
	return 1;
}

char RotateInvariantDeformer::fillDelta(const unsigned & numVertices, VertexAdjacency * adjacency)
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

void RotateInvariantDeformer::setMesh(BaseMesh * mesh)
{
	m_mesh = mesh;
	m_numVertices = mesh->getNumVertices();
	m_deformedV = new Vector3F[m_numVertices];
	
	printf("init laplace deformer");
	
	
	//fillDelta(m_numVertices, msh->connectivity());
}

char RotateInvariantDeformer::solve()
{
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_mesh);
	solveRotation(msh);
/*
	m_llt.solveInPlace(m_delta[0]);
	m_llt.solveInPlace(m_delta[1]);
	m_llt.solveInPlace(m_delta[2]);
	
	for(int i = 0; i < m_M.rows(); i++) {
		m_deformedV[i].x = m_delta[0](i);
		m_deformedV[i].y = m_delta[1](i);
		m_deformedV[i].z = m_delta[2](i);
	}*/
	return 1;
}

unsigned RotateInvariantDeformer::countEdges(const unsigned & numVertices, VertexAdjacency * adjacency)
{
    unsigned count = 0;
    for(int i = 0; i < (int)numVertices; i++) {
		VertexAdjacency & adj = adjacency[i];

		count += adj.getNumNeighbors();
	}
	return count;
}
