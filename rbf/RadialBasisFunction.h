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
class RadialBasisFunction {
public:
	RadialBasisFunction();
	virtual ~RadialBasisFunction();
	
	void setTau(float tau);
	void create(unsigned num);
	void setXi(unsigned idx, Vector3F xi);
	void setCi(unsigned idx, float ci);
	void computeWeights();
	float solve(Vector3F x) const;
private:
	float Phir(float r) const;
	Vector3F *m_xi;
	float *m_ci;
	float *m_wi;
	float m_tau;
	unsigned m_numNodes;
};