/*
 *  BaseFunction.h
 *  aphid
 *
 *  Created by jian zhang on 10/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
class BaseFunction {
public:
	BaseFunction();
	virtual ~BaseFunction();
	
	virtual unsigned ndim() const;
	virtual double f(const VectorN<double> & X);
	virtual double f(const Vector2F & x);
	
	void setLimit(float lo, float hi);
	float limitLow() const;
	float limitHigh() const;
	
private:
	float m_limit[2];
};