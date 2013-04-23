/*
 *  AnchorDeformer.h
 *  masq
 *
 *  Created by jian zhang on 3/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <LinearMath.h>
#include "BaseDeformer.h"
#include <MeshTopology.h>
#include <AnchorGroup.h>
#include <vector>

class VertexAdjacency;
class Matrix33F;

class AnchorDeformer : public BaseDeformer, public MeshTopology
{
public:
    AnchorDeformer();
    virtual ~AnchorDeformer();
	
	virtual void setMesh(BaseMesh * mesh);
	void setAnchors(AnchorGroup * ag);
	virtual void precompute();
	virtual char solve();
	
private:
	void prestep(Eigen::VectorXd b[], char isMembrane = 0);
	Matrix33F svdRotation(unsigned iv);
	LaplaceMatrixType m_L;
	LaplaceMatrixType m_LT;
	Eigen::VectorXd m_delta[3];
	Eigen::SimplicialLDLT<LaplaceMatrixType> m_llt;
	AnchorGroup * m_anchors;
	Vector3F * m_membrane;
};