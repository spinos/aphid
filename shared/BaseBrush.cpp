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
BaseBrush::BaseBrush() : m_radius(1.f), m_pitch(0.01f), m_maxToeFactor(2.f), m_numDarts(99)
{
	m_strength = 1.f;
	m_twoSided = false;
}

BaseBrush::~BaseBrush() 
{
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

void BaseBrush::setPitch(float x)
{
	m_pitch = x;
}

void BaseBrush::setStrength(float x)
{
	m_strength = x;
}

float BaseBrush::strength() const
{
	return m_strength;
}

void BaseBrush::setNumDarts(int x)
{
    m_numDarts = x;
}

void BaseBrush::setMaxToeFactor(float x)
{
	m_maxToeFactor = x;
}

Matrix44F BaseBrush::getSpace() const
{
	return m_space;
}

float BaseBrush::getRadius() const
{
	return m_radius;
}

float BaseBrush::getPitch() const
{
	return m_pitch;
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
	m_previousToeWorldP = m_toeWorldPos;
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

void BaseBrush::setToeByIntersect(const Ray * r, bool useNormal)
{
	Vector3F pn = normal();
	
	if(!useNormal) {
		pn = r->m_dir;
		pn.reverse();
		pn.normalize();
	}
	
	Plane pl(pn, heelPosition());
	Vector3F hit, d;
	float dd;
    float t;
    if(pl.rayIntersect(*r, hit, t)) {
        d = hit - heelPosition();
		dd = d.length();
		d /= dd;
		if(dd < minDartDistance()) 
			dd = minDartDistance();
		else if(dd > m_radius * m_maxToeFactor) 
			dd = m_radius * m_maxToeFactor;
		d *= dd;	
		
		m_toeWorldPos = heelPosition();
		m_toeWorldPos += d;
	}
}

const float BaseBrush::length() const
{
    return toeDisplacement().length();
}

const Vector3F BaseBrush::toeDisplacement() const
{
    return Vector3F(heelPosition(), toePosition());
}

const Vector3F BaseBrush::toeDisplacementDelta()
{
    Vector3F d0 = m_previousToeWorldP;
	Vector3F d1 = m_toeWorldPos;
	m_previousToeWorldP = m_toeWorldPos;
	return d1 - d0;
}

void BaseBrush::setTwoSided(bool b)
{
	m_twoSided = b;
}

bool BaseBrush::twoSided() const
{
	return m_twoSided;
}