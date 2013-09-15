/*
 *  BaseBrush.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseBrush.h"

BaseBrush::BaseBrush() : m_radius(1.f) 
{
	setNumDarts(8);
}

BaseBrush::~BaseBrush() 
{
	if(m_darts) delete[] m_darts;
}

void BaseBrush::setSpace(const Vector3F & point, const Vector3F & facing)
{
	m_space.setTranslation(point);
	m_space.setFrontOrientation(facing);
}

void BaseBrush::setRadius(float x)
{
	m_radius = x;
}

void BaseBrush::setNumDarts(unsigned x)
{
	if(m_darts) delete[] m_darts;
	m_numDarts = x;
	m_darts = new Vector3F[m_numDarts];
	
	const float fac = .6f/m_numDarts;
	unsigned valid = 0;
	while (valid < m_numDarts) {
		const float alpha = (rand()%131/131.f*2.f - 1.f) * PI;
		const float r = rand()%143/143.f;
		Vector3F p(r * cos(alpha), r * sin(alpha), 0.f);
		if(!ignoreTooClose(p, m_darts, valid, m_radius * fac)) {
			m_darts[valid] = p;
			valid++;
		}
	}
}

Matrix44F BaseBrush::getSpace() const
{
	return m_space;
}

float BaseBrush::getRadius() const
{
	return m_radius;
}

unsigned BaseBrush::getNumDarts() const
{
	return m_numDarts;
}

Ray BaseBrush::getObjectRay(unsigned idx) const
{
	Vector3F ori = m_darts[idx] * m_radius;
	Vector3F dst = ori;
	ori.z = m_radius;
	ori = m_space.transform(ori);
	
	dst.z = - m_radius;
	dst = m_space.transform(dst);
	
	return Ray(ori, dst);
}

char BaseBrush::ignoreTooClose(Vector3F p, Vector3F *data, unsigned count, float d) const
{
	for(unsigned i = 0; i <count; i++) {
		Vector3F v = p - data[i];
		if(v.length() < d) return 1;
	}
	return 0;
}