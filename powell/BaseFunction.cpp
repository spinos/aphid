/*
 *  BaseFunction.cpp
 *  aphid
 *
 *  Created by jian zhang on 10/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseFunction.h"
#include <iostream>
BaseFunction::BaseFunction() {}
BaseFunction::~BaseFunction() {}

float BaseFunction::f(const Vector2F & x)
{
	//return cos(x.x) + sin(x.y);//f(3.14162,4.71236)=-2
	//return 100.0 * (x.y - x.x * x.x) * (x.y - x.x * x.x) + (1.0 - x.x) * (1.0 - x.x);// failed
	return x.x * x.x - 4 * x.x + x.y * x.y - x.y - x.x * x.y;//f(3, 2)=-7
	return 3 + (x.x - 1.5 * x.y) * (x.x - 1.5 * x.y) + (x.y - 2.f) * (x.y - 2.f);//f(3, 2)=3
}

float BaseFunction::f1(float x, const Vector2F & at, const Vector2F & S)
{
	return f(Vector2F(at.x + x * S.x, at.y + x * S.y));
}

float BaseFunction::f2(const Vector2F & x, const Vector2F & at, const Vector2F & S)
{
	//return f(Vector2F(at.x + x.x * S.x, at.y + x.y * S.y));
	return 3 + (at.x - 1.5 * (at.y + x.y * S.y)) * (at.x - 1.5 * (at.y + x.y * S.y)) + (at.y + x.y * S.y - 2.f) * (at.y + x.y * S.y - 2.f);
}

float BaseFunction::particalDerivativeAt(float x, const Vector2F & at, const Vector2F & S)
{
	return (f(at + S * (x + 0.1f)) - f(at + S * (x - 0.1f))) / 0.2f;
}

float BaseFunction::minimization(const Vector2F & x, const Vector2F & part)
{
	float r = 0.f;
	
	float d = particalDerivativeAt(r, x, part); 
	char isCritical = (d * d < 10e-11);
	int i = 0;
	while(!isCritical) {
		r -= d * .1f;
		
		d = particalDerivativeAt(r, x, part);
		isCritical = (d * d < 10e-11);
		i++;
	}
	
	//std::cout<<"n step "<<i<<"\n";
	return r;
}