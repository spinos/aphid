#pragma once
#include <LinearMath.h>
#include "BaseField.h"
#include "Matrix33F.h"
#include <vector>
#include <map>

class Anchor;
class VertexAdjacency;

class HarmonicCoord : public BaseField
{
public:
    HarmonicCoord();
    virtual ~HarmonicCoord();
	
	virtual void setMesh(BaseMesh * mesh);
	void setTopology(VertexAdjacency * topo);
	virtual void precompute(std::vector<Anchor *> & anchors);
	virtual char solve(unsigned iset);
	
	unsigned numAnchorPoints() const;
	
	void setConstrain(unsigned idx, float val);
private:
	void initialCondition();
	void prestep();
	void checkConstrain(unsigned iset);

	std::vector<Anchor *> m_anchors;
	LaplaceMatrixType m_LT;
	Eigen::VectorXd m_b;
	Eigen::SimplicialLDLT<LaplaceMatrixType> m_llt;
	VertexAdjacency * m_topology;
	float * m_constrainValues;
	unsigned m_numAnchors;
};
