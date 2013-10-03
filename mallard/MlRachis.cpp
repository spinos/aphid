/*
 *  MlRachis.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/20/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlRachis.h"
#include <CollisionRegion.h>

MlRachis::MlRachis() : m_spaces(0), m_angles(0) {}
MlRachis::~MlRachis() 
{
	if(m_spaces) delete[] m_spaces;
	if(m_angles) delete[] m_angles;
}

void MlRachis::create(unsigned x)
{
	if(m_spaces) delete[] m_spaces;
	if(m_angles) delete[] m_angles;
	m_numSpace = x;
	m_spaces = new Matrix33F[x];
	m_angles = new float[x];
}

void MlRachis::computeAngles(float * segL, float fullL)
{
	for(unsigned i = 0; i < m_numSpace; i++) {
		const float fac = (float)i/(float)m_numSpace;
		m_angles[i] = segL[i] / fullL * (1.f - fac) + sqrt(segL[i] / fullL) * fac;
		m_angles[i] *= 1.f + fac;
	}
}

void MlRachis::reset()
{
	for(unsigned i = 0; i < m_numSpace; i++) m_spaces[i].setIdentity(); 
}

void MlRachis::update(const Vector3F & oriP, const Matrix33F & space, const float & scale, CollisionRegion * collide, const float & fullPitch)
{
	Vector3F zdir(0.f, 0.f, scale);
	zdir = space.transform(zdir);
	Vector3F xdir(1.f, 0.f, 0.f);
	xdir = space.transform(xdir);
	
	const Vector3F toP = oriP + zdir;
	const Vector3F clsP = collide->getClosestPoint(toP);
	const Vector3F clsV(oriP, clsP);
	
	const float bounceAngle = zdir.angleBetween(clsV, xdir);
	
	reset();
	
	if(bounceAngle > 0.1) m_spaces[0].rotateY(fullPitch * m_angles[0] + bounceAngle);
	else m_spaces[0].rotateY(fullPitch * m_angles[0]);
	for(unsigned i = 1; i < m_numSpace; i++) {
		m_spaces[i].rotateY((fullPitch + bounceAngle) * m_angles[i]);
	}
}

Matrix33F MlRachis::getSpace(short idx) const
{
	return m_spaces[idx];
}