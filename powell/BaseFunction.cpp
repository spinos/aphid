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
BaseFunction::BaseFunction() 
{
	m_limit[0] = -10;
	m_limit[1] = 10;
}

BaseFunction::~BaseFunction() {}

unsigned BaseFunction::ndim() const
{
	return 2;
}

double BaseFunction::f(const VectorN<double> & X)
{
	//return cos(X[0]) + sin(X[1]);//f(3.14162,4.71236)=-2
	return X[0] * X[0] - 4 * X[0] + X[1] * X[1] - X[1] - X[0] * X[1];//f(3, 2)=-7
	return 3 + (X[0] - 1.5 * X[1]) * (X[0] - 1.5 * X[1]) + (X[1] - 2) * (X[1] - 2);
}

double BaseFunction::f(const Vector2F & x)
{
	//return cos(x.x) + sin(x.y);//f(3.14162,4.71236)=-2
	//return 100.0 * (x.y - x.x * x.x) * (x.y - x.x * x.x) + (1.0 - x.x) * (1.0 - x.x);// failed
	//return x.x * x.x - 4 * x.x + x.y * x.y - x.y - x.x * x.y;//f(3, 2)=-7
	return 3 + (x.x - 1.5 * x.y) * (x.x - 1.5 * x.y) + (x.y - 2.f) * (x.y - 2.f);//f(3, 2)=3
}

void BaseFunction::setLimit(float lo, float hi)
{
	m_limit[0] = lo;
	m_limit[1] = hi;
}

float BaseFunction::limitLow() const
{
	return m_limit[0];
}

float BaseFunction::limitHigh() const
{
	return m_limit[1];
}
