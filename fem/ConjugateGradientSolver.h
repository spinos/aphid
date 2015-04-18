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
class BaseBuffer;
class CUDABuffer;
class CudaCSRMatrix;
typedef std::map<int, Matrix33F> MatrixMap;

class ConjugateGradientSolver {
public:
	ConjugateGradientSolver();
	virtual ~ConjugateGradientSolver();
	
	virtual void initOnDevice();
	
	void init(unsigned n);
	void solve(Vector3F * X); 
	void solveGpu(Vector3F * X, CudaCSRMatrix * stiffnessMatrix);
	
	int * isFixed();
	Vector3F * rightHandSide();
	Matrix33F * A(unsigned i, unsigned j);
private:
	MatrixMap * m_A_row;
	Vector3F * m_b;
	Vector3F * m_residual;
	Vector3F * m_update;
	BaseBuffer * m_prev;
	int * m_IsFixed;
	CUDABuffer * m_deviceIsFixed;
	CUDABuffer * m_deviceResidual;
	CUDABuffer * m_deviceUpdate;
	CUDABuffer * m_devicePrev;
	CUDABuffer * m_deviceD;
	CUDABuffer * m_deviceD2;
	CUDABuffer * m_deviceX;
	CUDABuffer * m_deviceRhs;
	BaseBuffer * m_hostD;
	BaseBuffer * m_hostD2;
	unsigned m_numRows;
};