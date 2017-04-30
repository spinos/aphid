/*
 *  LaplaceSmoother.h
 *  mallard
 *
 *  Created by jian zhang on 12/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <AllMath.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> LaplaceMatrixType;
class MeshTopology;
class LaplaceSmoother {
public:
	LaplaceSmoother();
	virtual ~LaplaceSmoother();
	
	void precompute(unsigned num, MeshTopology * topo, const std::vector<unsigned> & constraintIdx, const std::vector<float> & constraintWei);
	
	void cleanup();
	void solve(Vector3F * entry);
protected:

private:
	unsigned numRows() const;
	void computeRightHandSide(Vector3F * entry);
	MeshTopology * m_topology;
	LaplaceMatrixType m_LT;
	Eigen::VectorXf m_delta[3];
	Eigen::SparseLLT<LaplaceMatrixType> m_llt;
	unsigned m_numData, m_numConstraint;
	unsigned * m_constraintIndices;
	float * m_constraintWeights;
};
