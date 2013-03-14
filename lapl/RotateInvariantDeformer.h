#pragma once
#include "LaplaceDeformer.h"

class VertexAdjacency;
class RotateInvariantDeformer : public BaseDeformer
{
public:
    RotateInvariantDeformer();
    virtual ~RotateInvariantDeformer();
	
	virtual void setMesh(BaseMesh * mesh);
	
	virtual char solve();
	
	char fillM(const unsigned & numVertices, VertexAdjacency * adjacency);
	char fillDelta(const unsigned & numVertices, VertexAdjacency * adjacency);
private:
    unsigned countEdges(const unsigned & numVertices, VertexAdjacency * adjacency);
    LaplaceMatrixType m_M, m_LT;
	Eigen::VectorXf m_delta[3];
	Eigen::SparseLLT<LaplaceMatrixType> m_llt;
};
