/*
 *  PowellMethod.h
 *  aphid
 *
 *  Created by jian zhang on 10/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BaseFunction.h>

class PowellMethod {
public:
	PowellMethod();
	
	void solve(BaseFunction & F, Vector2F & x);
private:
	void cycle(BaseFunction & F, Vector2F & x, Vector2F & S0, Vector2F & S1);
	double taxiCab(double x);
	double goldenSectionSearch(double a, double b, double c, double tau);
	double minimization(BaseFunction & F, const Vector2F & at, const Vector2F & along);
	
	BaseFunction * m_f;
	Vector2F m_at, m_along;
};