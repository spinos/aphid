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
#include <Patch.h>

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
		m_angles[i] = sqrt(segL[i] / fullL) * .5f + segL[i] / fullL * .5f;
	}
}

void MlRachis::reset()
{
	for(unsigned i = 0; i < m_numSpace; i++) m_spaces[i].setIdentity(); 
}

void MlRachis::update(const Vector3F & oriP, const Matrix33F & space, float radius, CollisionRegion * collide, const float & fullPitch)
{
	Vector3F zdir(0.f, 0.f, 1.f);
	zdir = space.transform(zdir);
	Vector3F xdir(1.f, 0.f, 0.f);
	xdir = space.transform(xdir);
	
	Patch::PushPlaneContext ctx;
	ctx.reset(xdir, oriP, zdir, radius);
	collide->pushPlane(&ctx);
	float bounceAngle = ctx.m_maxAngle;
	
	reset();
	
	for(unsigned i = 0; i < m_numSpace; i++) {
		m_spaces[i].rotateY((fullPitch + bounceAngle) * m_angles[i]);
	}
}

Matrix33F MlRachis::getSpace(short idx) const
{
	return m_spaces[idx];
}