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
	
	unsigned getNumAnchors() const;

private:
	void initialCondition();
	char fillL();
	char fillDelta();
	void updateRi();
    LaplaceMatrixType m_LT;
	Eigen::VectorXf m_delta[3];
	Eigen::SparseLLT<LaplaceMatrixType> m_llt;
	Vector3F *m_ViVjWj;
	Eigen::Matrix3f *m_mRi;
	VertexAdjacency * m_topology;
};
