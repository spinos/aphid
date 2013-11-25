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

MlRachis::MlRachis() : m_spaces(0), m_angles(0), m_lengthPortions(0) {}
MlRachis::~MlRachis() 
{
	if(m_spaces) delete[] m_spaces;
	if(m_angles) delete[] m_angles;
	if(m_lengthPortions) delete[] m_lengthPortions;
}

void MlRachis::create(unsigned x)
{
	if(m_spaces) delete[] m_spaces;
	if(m_angles) delete[] m_angles;
	if(m_lengthPortions) delete[] m_lengthPortions;
	m_numSpace = x;
	m_spaces = new Matrix33F[x];
	m_angles = new float[x];
	m_lengthPortions = new float[x];
}

void MlRachis::computeAngles(float * segL, float fullL)
{
    for(unsigned i = 0; i < m_numSpace; i++) {
		m_lengthPortions[i] = segL[i] / fullL;
		m_angles[i] = m_lengthPortions[i] * .8f + m_lengthPortions[i] * m_lengthPortions[i] * .2f;
	}
}

void MlRachis::reset()
{
	for(unsigned i = 0; i < m_numSpace; i++) m_spaces[i].setIdentity(); 
}

void MlRachis::bend(const Vector3F & oriP, const Matrix33F & space, float radius, CollisionRegion * collide, const float & fullPitch)
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
	
	Matrix33F localSpace, invSpace, segSpace = space;
	m_spaces[0].rotateY(.1f + bounceAngle);
	
	Vector3F localD, localU, segU, segP = oriP;
	float segRot;
	
	localSpace = m_spaces[0];
	localSpace.multiply(segSpace);
	
	localD.set(0.f, 0.f, radius * m_lengthPortions[0]);
	localD = localSpace.transform(localD);
		
	segP += localD;
	
	segSpace = localSpace;
	float pitchPortion = 0.f;
	for(unsigned i = 1; i < m_numSpace - 2; i++) {
		invSpace = segSpace;
		invSpace.inverse();

		segU = collide->getClosestNormal(segP);
		segU = invSpace.transform(segU);
		segU.y = 0.f;
		
		segRot = acos(segU.normal().dot(Vector3F::XAxis));
		if(segU.z > 0.f) segRot *= -1.f;

		pitchPortion += m_angles[i];
		m_spaces[i].rotateY(bounceAngle * m_lengthPortions[0] + fullPitch * pitchPortion + segRot);

		localSpace = m_spaces[i];
		localSpace.multiply(segSpace);
		
		localD.set(0.f, 0.f, radius * m_lengthPortions[i]);
		localD = localSpace.transform(localD);
		
		segP += localD;
		
		segSpace = localSpace;
	}
	
	for(unsigned i = m_numSpace - 2; i < m_numSpace; i++) {
	    pitchPortion += m_angles[i];
	    m_spaces[i].rotateY(fullPitch * pitchPortion);
	}
}

Matrix33F MlRachis::getSpace(short idx) const
{
	return m_spaces[idx];
}