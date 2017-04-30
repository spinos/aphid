/*
 *  PowellMethod.h
 *  aphid
 *
 *  Created by jian zhang on 10/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <BaseFunction.h>

class PowellMethod {
public:
	PowellMethod();
	
	void solve(BaseFunction & F, VectorN<double> & x);
private:
	void cycle(BaseFunction & F, VectorN<double> & x, VectorN<double> * U);
	double taxiCab(double x);
	double goldenSectionSearch(double a, double b, double c, double tau);
	double minimization(BaseFunction & F, const VectorN<double> & at, const VectorN<double> & along);
	
	BaseFunction * m_f;
	VectorN<double> m_at; 
	VectorN<double> m_along;
};