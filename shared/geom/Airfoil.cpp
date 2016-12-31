/*
 *  Airfoil.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 12/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Airfoil.h"
#include <cmath>

namespace aphid {

/// 2415
Airfoil::Airfoil() :
m_c(1.f),
m_m(.02f),
m_p(.4f),
m_t(.15f)
{}

Airfoil::Airfoil(const float & c,
			const int & m,
			const int & p,
			const int & t1,
			const int & t2)
{ 
	setChord(c);
	set4Digit(m, p, t1, t2);
}

Airfoil::~Airfoil()
{}

void Airfoil::setChord(const float & c)
{ m_c = c; }

void Airfoil::set4Digit(const int & m,
					const int & p,
					const int &t1,
					const int & t2)
{
	m_m = .01f * m * m_c;
	m_p = .1f * p * m_c;
	m_t = (.1f * t1 + .01f * t2) * m_c;
}

const float & Airfoil::chord() const
{ return m_c; }

float Airfoil::calcYc(const float & x) const
{
	float Yc = 0.f;
	if(m_m < 1.e-5f) {
		return Yc;
	}
	if(x < m_p) {
		Yc = m_m * (2.f * m_p * x - x * x) / (m_p * m_p);
	} else {
		Yc = m_m * (1.f - 2.f * m_p + 2.f * m_p * x - x * x) / ((1.f - m_p) * (1.f - m_p) ); 
	}
	return Yc;
}

float Airfoil::calcYt(const float & x) const
{
	return (m_t * (sqrt(x) * .2969f - x * .126f
				- x * x * .3516f
				+ x * x * x * .2843f
				- x * x * x * x * .1015f) / .2f);
}

float Airfoil::calcTheta(const float & x) const
{
	if(m_m < 1.e-5f) {
		return 0.f;
	}
	
	const float dx = 1.e-3f * m_c;
	float x0 = x;
	float x1 = x + dx;
	if(x1 > m_c) {
		x0 = m_c - dx;
		x1 = m_c;
	}
	
	float yc0 = calcYc(x0);
	float yc1 = calcYc(x1);
	
	return atan((yc1 - yc0) / dx);
}

}
