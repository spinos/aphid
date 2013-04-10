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
	virtual void addTargetAnalysis(DeformationTarget * analysis);
	virtual void precompute();
	virtual char solve();
	
private:
	void prestep(Eigen::VectorXd b[]);
	void addupConstrains(DeformationTarget * target, Eigen::VectorXd b[]);
	LaplaceMatrixType m_L;
	Eigen::VectorXd m_delta[3];
	Eigen::SimplicialLDLT<LaplaceMatrixType> m_llt;
	VertexAdjacency * m_topology;
	std::vector<DeformationTarget *> m_analysis;
};