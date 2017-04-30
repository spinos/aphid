/*
 *  QuadraticCurve.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 5/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "QuadraticCurve.h"
namespace aphid {

QuadraticCurve::QuadraticCurve() {}
QuadraticCurve::~QuadraticCurve() {}

Vector3F QuadraticCurve::interpolate(float param) const
{
	unsigned k0 = 0;
	unsigned k1 = numVertices() - 1;
	
	if(param <= 0.f) return m_cvs[k0];
	if(param >= 1.f) return m_cvs[k1];
	
	findNeighborKnots(param, k0, k1);
	
	return interpolateStraight(param - m_knots[k0], k0, k1);
}

}