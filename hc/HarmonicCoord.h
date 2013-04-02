#pragma once
#include "BaseField.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Matrix33F.h"
#include "Anchor.h"
#include <vector>
#include <map>
typedef Eigen::SparseMatrix<float> LaplaceMatrixType;

class VertexAdjacency;

class HarmonicCoord : public BaseField
{
public:
    HarmonicCoord();
    virtual ~HarmonicCoord();
	
	virtual void setMesh(BaseMesh * mesh);
	virtual void precompute(std::vector<Anchor *> & anchors);
	virtual char solve();
	
	unsigned numAnchorPoints() const;

private:
	void initialCondition();
	void prestep();
	std::map<unsigned, Anchor::AnchorPoint *> m_anchorPoints;
    LaplaceMatrixType m_LT;
	Eigen::VectorXf m_b;
	Eigen::SimplicialLDLT<LaplaceMatrixType> m_llt;
	VertexAdjacency * m_topology;
};
