/*
 *  TransferDeformer.h
 *  lapl
 *
 *  Created by jian zhang on 3/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

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
class DeformationAnalysis;
class TransferDeformer : public BaseDeformer
{
public:
    TransferDeformer();
    virtual ~TransferDeformer();
	
	virtual void setMesh(BaseMesh * mesh);
	virtual void setTargetAnalysis(DeformationAnalysis * analysis);
	virtual void setBaseAnalysis(DeformationAnalysis * analysis);
	virtual void precompute(std::vector<Anchor *> & anchors);
	virtual char solve();
	
	unsigned numAnchorPoints() const;

private:
	void prestep();
	
	std::map<unsigned, Anchor::AnchorPoint *> m_anchorPoints;
    LaplaceMatrixType m_LT;
	Eigen::VectorXf m_delta[3];
	Eigen::SparseLLT<LaplaceMatrixType> m_llt;
	VertexAdjacency * m_topology;
	DeformationAnalysis * m_targetAnalysis;
	DeformationAnalysis * m_baseAnalysis;
};