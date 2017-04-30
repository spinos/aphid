#pragma once
#include "LaplaceDeformer.h"
class MeshLaplacian;
class VertexAdjacency;
class RotateInvariantDeformer : public BaseDeformer
{
public:
    RotateInvariantDeformer();
    virtual ~RotateInvariantDeformer();
	
	virtual void setMesh(BaseMesh * mesh);
	
	virtual char solve();
	
	char fillMRot(VertexAdjacency * adjacency, LaplaceMatrixType &ATA, LaplaceMatrixType &AT);
	char solveRotation(MeshLaplacian *msh);
	char fillDelta(const unsigned & numVertices, VertexAdjacency * adjacency);
private:
    unsigned countEdges(const unsigned & numVertices, VertexAdjacency * adjacency);
    LaplaceMatrixType m_M, m_LT;
	LaplaceMatrixType m_MRot;
	Eigen::VectorXf m_delta[3];
	Eigen::SparseLLT<LaplaceMatrixType> m_llt;
};
