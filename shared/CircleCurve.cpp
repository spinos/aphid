/*
 *  CircleCurve.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "CircleCurve.h"
namespace aphid {

CircleCurve::CircleCurve() : m_radius(1.f), m_eccentricity(0.f)
{   
	create();
}

void CircleCurve::create()
{
	createVertices(37);
    setRadius(1.f);
}

CircleCurve::~CircleCurve() {}

void CircleCurve::setRadius(float x)
{
	m_radius = x;
	const float delta = PI / 18.f;
	float a, b, r;
    for(int i = 0; i < 37; i++) {
		if(m_eccentricity > 0.f && m_eccentricity < 1.f) {
			a = m_radius;
			b = sqrt( 1.f - m_eccentricity * m_eccentricity) * a;
			r = a * b / sqrt(a * a * cos(delta * i) * cos(delta * i) + b * b * sin(delta * i) * sin(delta * i));
		}
		else
			r = m_radius;
		
		m_cvs[i].set(sin(delta * i) * r, cos(delta * i) * r, 0.f);
	}
	computeKnots();
}

}