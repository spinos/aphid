/*
 *  ConjugateGradientSolver.h
 *  fem
 *
 *  Created by jian zhang on 1/8/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <map>
typedef std::map<int, Matrix33F> MatrixMap;

class ConjugateGradientSolver {
public:
	ConjugateGradientSolver();
	virtual ~ConjugateGradientSolver();
	void init(unsigned n);
	void solve(Vector3F * X); 
	
	bool * isFixed();
	Vector3F * rightHandSide();
	Matrix33F * A(unsigned i, unsigned j);
private:
	MatrixMap * m_A_row;
	Vector3F * m_b;
	Vector3F * m_residual;
	Vector3F * m_update;
	Vector3F * m_prev;
	bool * m_IsFixed;
	
	unsigned m_numRows;
};