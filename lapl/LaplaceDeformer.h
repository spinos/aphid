#pragma once
#include "BaseDeformer.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Matrix33F.h"
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> LaplaceMatrixType;

class VertexAdjacency;
class LaplaceDeformer : public BaseDeformer
{
public:
    LaplaceDeformer();
    virtual ~LaplaceDeformer();
	
	virtual void setMesh(BaseMesh * mesh);
	
	virtual char solve();
	
	void initialCondition();
	char fillM();
	char fillDelta();
	void LaplaceDeformer::updateRi();
private:
    LaplaceMatrixType m_M, m_LT;
	Eigen::VectorXf m_delta[3];
	Eigen::SparseLLT<LaplaceMatrixType> m_llt;
	Vector3F *m_vPi;
	Matrix33F *m_mRi;
	VertexAdjacency * m_topology;
};
