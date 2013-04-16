/*
 *  RadialBasisFunction.h
 *  rbf
 *
 *  Created by jian zhang on 4/12/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <AllMath.h>
#include <LinearMath.h>
#include <vector>
class RadialBasisFunction {
public:
	RadialBasisFunction();
	virtual ~RadialBasisFunction();
	
	void setTau(float tau);
	void create(unsigned num);
	void setXi(unsigned idx, Vector3F xi);
	void computeWeights();
	void solve(Vector3F x) const;
	
	unsigned getNumNodes() const;
	Vector3F getXi(unsigned idx) const;
	float getResult(unsigned idx) const;
private:
	float Phir(float r) const;
	void weightI(unsigned idx, Eigen::MatrixXd & A);
	Vector3F *m_xi;
	std::vector<float *> m_ci;
	std::vector<float *> m_wi;
	float * m_res;
	float m_tau;
	unsigned m_numNodes;
};