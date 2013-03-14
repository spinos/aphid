#include "RotateInvariantDeformer.h"
#include "VertexAdjacency.h"
#include "MeshLaplacian.h"
RotateInvariantDeformer::RotateInvariantDeformer() {}
RotateInvariantDeformer::~RotateInvariantDeformer() {}

char RotateInvariantDeformer::fillM(const unsigned & numVertices, VertexAdjacency * adjacency)
{
    MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_mesh);
    const unsigned numEdges = countEdges(numVertices, adjacency);
    
	std::map<int,int> ordered;
	std::map<int,int>::iterator orderIt;
	int neighborIdx;
	float neighborWei;
	unsigned i3, j3;
	
	LaplaceMatrixType L(numEdges * 3, numVertices * 3);
	L.setZero();
	L.startFill();
	
	unsigned row = 0;
	for(int i = 0; i < (int)numVertices; i++) {
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
			    if(j3 < i3)
				    L.fill( row+k, j3+k ) = neighborWei * -1.0f;
				L.fill( row+k, i3+0 ) = neighborWei * Rij(k, 0);
				L.fill( row+k, i3+1 ) = neighborWei * Rij(k, 1);
				L.fill( row+k, i3+2 ) = neighborWei * Rij(k, 2);
				if(j3 > i3)
				    L.fill( row+k, j3+k ) = neighborWei * -1.0f;
			}
			row += 3; 
		}
	}
	L.endFill();
	
	m_LT = L.transpose();
	m_M = m_LT * L;
	//std::cout << "M \n" << m_M << std::endl;
	m_llt = Eigen::SparseLLT<LaplaceMatrixType>(m_M);
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
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_mesh);

	fillM(m_numVertices, msh->connectivity());
	fillDelta(m_numVertices, msh->connectivity());
}

char RotateInvariantDeformer::solve()
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

unsigned RotateInvariantDeformer::countEdges(const unsigned & numVertices, VertexAdjacency * adjacency)
{
    unsigned count = 0;
    for(int i = 0; i < (int)numVertices; i++) {
		VertexAdjacency & adj = adjacency[i];

		count += adj.getNumNeighbors();
	}
	return count;
}
