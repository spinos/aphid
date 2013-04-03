/*
 *  AccumulateDeformer.h
 *  lapl
 *
 *  Created by jian zhang on 3/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <LinearMath.h>
#include "BaseDeformer.h"
#include "Matrix33F.h"
#include "Anchor.h"
#include <vector>
#include <map>

class VertexAdjacency;
class DeformationTarget;
class AccumulateDeformer : public BaseDeformer
{
public:
    AccumulateDeformer();
    virtual ~AccumulateDeformer();
	
	virtual void setMesh(BaseMesh * mesh);
	virtual void setTargetAnalysis(DeformationTarget * analysis);
	virtual void setBaseAnalysis(DeformationTarget * analysis);
	virtual void precompute(std::vector<Anchor *> & anchors);
	virtual char solve();
	
	unsigned numAnchorPoints() const;

private:
	void prestep();
	
	std::map<unsigned, Anchor::AnchorPoint *> m_anchorPoints;
    LaplaceMatrixType m_LT;
	Eigen::VectorXf m_delta[3];
	Eigen::SimplicialLDLT<LaplaceMatrixType> m_llt;
	VertexAdjacency * m_topology;
	DeformationTarget * m_targetAnalysis;
	DeformationTarget * m_baseAnalysis;
};