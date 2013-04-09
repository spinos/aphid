#pragma once
#include <LinearMath.h>
#include "BaseField.h"
#include "Matrix33F.h"
#include "WeightHandle.h"
#include <vector>
#include <map>

class VertexAdjacency;
//class ControlGraph;

class HarmonicCoord : public BaseField
{
public:
    HarmonicCoord();
    virtual ~HarmonicCoord();
	
	virtual void setMesh(BaseMesh * mesh);
	virtual void precompute(std::vector<WeightHandle *> & anchors);
	virtual char solve(unsigned iset);
	
	unsigned numAnchorPoints() const;
	bool hasNoEffect() const;
	
	void setConstrain(unsigned idx, float val);
private:
	void initialCondition();
	void prestep();
	bool allZero() const;
	std::vector<WeightHandle *> m_anchors;
	//ControlGraph * m_controls;
	LaplaceMatrixType m_LT;
	Eigen::VectorXf m_b;
	Eigen::SimplicialLDLT<LaplaceMatrixType> m_llt;
	VertexAdjacency * m_topology;
	float * m_constrainValues;
	unsigned m_numAnchors;
};
