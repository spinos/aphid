/*
 *  MembraneDeformer.h
 *  masq
 *
 *  Created by jian zhang on 3/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <LinearMath.h>
#include "BaseDeformer.h"
#include <AnchorGroup.h>
#include <vector>

class VertexAdjacency;
class HarmonicCoord;
class Matrix33F;

class MembraneDeformer : public BaseDeformer
{
public:
    MembraneDeformer();
    virtual ~MembraneDeformer();
	
	virtual void setMesh(BaseMesh * mesh);
	void setAnchors(AnchorGroup * ag);
	virtual void precompute();
	virtual char solve();
	
private:
	char buildTopology();
	void prestep(Eigen::VectorXd b[]);

	LaplaceMatrixType m_L;
	LaplaceMatrixType m_LT;
	Eigen::VectorXd m_delta[3];
	Eigen::SimplicialLDLT<LaplaceMatrixType> m_llt;
	VertexAdjacency * m_topology;
	AnchorGroup * m_anchors;
};