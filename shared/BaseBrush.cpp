/*
 *  BaseBrush.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseBrush.h"
#include <Plane.h>
BaseBrush::BaseBrush() : m_radius(3.f) 
{
	setNumDarts(25);
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
	
	const float minD = 1.f / sqrt((float)m_numDarts);
	unsigned valid = 0;
	while (valid < m_numDarts) {
		const float alpha = (rand()%131/131.f*2.f - 1.f) * PI;
		const float r = rand()%143/143.f;
		Vector3F p(r * cos(alpha), r * sin(alpha), 0.f);
		if(!ignoreTooClose(p, m_darts, valid, minD)) {
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
	
	dst.z = -m_radius;
	dst = m_space.transform(dst);
	
	return Ray(ori, dst);
}

void BaseBrush::getDartPoint(unsigned idx, Vector3F & p) const
{
    p = m_darts[idx] * m_radius;
    p = m_space.transform(p);
}

char BaseBrush::ignoreTooClose(Vector3F p, Vector3F *data, unsigned count, float d) const
{
	for(unsigned i = 0; i <= count; i++) {
		Vector3F v = p - data[i];
		if(v.length() < d) return 1;
	}
	return 0;
}

float BaseBrush::minDartDistance() const
{
	return m_radius / sqrt((float)m_numDarts);
}

void BaseBrush::resetToe()
{
    m_toeWorldPos = heelPosition();
}

const Vector3F BaseBrush::heelPosition() const
{
    return m_space.getTranslation();
}

const Vector3F BaseBrush::toePosition() const
{
    return m_toeWorldPos;
}

const Vector3F BaseBrush::normal() const
{
    return m_space.getFront();
}

void BaseBrush::setToeByIntersectNormal(const Ray * r)
{
    Plane pl(normal(), heelPosition());
    Vector3F hit;
    float t;
    if(pl.rayIntersect(*r, hit, t))
        m_toeWorldPos = hit;
}

const float BaseBrush::length() const
{
    return toeDisplacement().length();
}

const Vector3F BaseBrush::toeDisplacement() const
{
    return Vector3F(heelPosition(), toePosition());
}
