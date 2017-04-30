#pragma once
#include "BaseDeformer.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Matrix33F.h"
#include "Anchor.h"
#include <vector>
#include <map>
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> LaplaceMatrixType;

class VertexAdjacency;

class LaplaceDeformer : public BaseDeformer
{
public:
    LaplaceDeformer();
    virtual ~LaplaceDeformer();
	
	virtual void setMesh(BaseMesh * mesh);
	virtual void precompute(std::vector<Anchor *> & anchors);
	virtual char solve();
	
	unsigned numAnchorPoints() const;

private:
	void initialCondition();
	void prestep();
	char fillL();
	char fillDelta();
	void updateRi();
	std::map<unsigned, Anchor::AnchorPoint *> m_anchorPoints;
    LaplaceMatrixType m_LT;
	Eigen::VectorXf m_delta[3];
	Eigen::SparseLLT<LaplaceMatrixType> m_llt;
	//Vector3F *m_ViVjWj;
	Eigen::Matrix3f *m_mRi;
	VertexAdjacency * m_topology;
};
