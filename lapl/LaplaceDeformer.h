#pragma once
#include "BaseDeformer.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<float, Eigen::RowMajor> LaplaceMatrixType;

class VertexAdjacency;
class LaplaceDeformer : public BaseDeformer
{
public:
    LaplaceDeformer();
    virtual ~LaplaceDeformer();
	
	virtual char solve();
	
	char fillM(const unsigned & numVertices, VertexAdjacency * adjacency);
	char fillDelta(const unsigned & numVertices, VertexAdjacency * adjacency);
private:
    LaplaceMatrixType m_M, m_LT;
	Eigen::VectorXf m_delta[3];
	Eigen::SparseLLT<LaplaceMatrixType> m_llt;
};
