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
	float acc = 0.f;
    for(unsigned i = 0; i < m_numSpace; i++) {
		m_lengthPortions[i] = segL[i] / fullL;
		acc += m_lengthPortions[i];
		m_angles[i] = acc;// * acc * 0.5f + acc * 0.5f;
	}
}

void MlRachis::reset()
{
	for(unsigned i = 0; i < m_numSpace; i++) m_spaces[i].setIdentity(); 
}

void MlRachis::bend(const Vector3F & oriP, const Matrix33F & space, float radius, CollisionRegion * collide, const float & fullPitch)
{
	reset();
	
	Vector3F zdir(0.f, 0.f, 1.f);
	zdir = space.transform(zdir);
	Vector3F xdir(1.f, 0.f, 0.f);
	xdir = space.transform(xdir);
	
	Matrix33F localSpace, invSpace, segSpace = space;
	Vector3F localD, localU, pop, topop, segU, segP = oriP;
	float segRot, pushAngle, curAngle;

	for(unsigned i = 0; i < m_numSpace; i++) {
		invSpace = segSpace;
		invSpace.inverse();

		localD.set(0.f, 0.f, 1.f);
		localD = segSpace.transform(localD);
		localD.normalize();
		
		localU.set(1.f, 0.f, 0.f);
		localU = segSpace.transform(localU);
		localU.normalize();
		
		segU = collide->getClosestNormal(segP + localD * radius * m_lengthPortions[i], radius, pop);
		
		topop = pop - segP;
		topop = invSpace.transform(topop);
		if(topop.x > 0.f) {
			topop.y = 0.f;
			pushAngle = acos(Vector3F::ZAxis.dot(topop.normal()));
		}
		else
			pushAngle = 0.f;
		
		segU = invSpace.transform(segU);
		segU.y = 0.f;
		
		segRot = acos(segU.normal().dot(Vector3F::XAxis));
		if(segU.z > 0.f) segRot *= -1.f;
		
		if(pushAngle > segRot && pushAngle > 0.f) curAngle = pushAngle;
		else curAngle = segRot * (1.f - fullPitch);
		
		curAngle += fullPitch + 0.5f * m_lengthPortions[i];
		
		m_spaces[i].rotateY(curAngle);

		localSpace = m_spaces[i];
		localSpace.multiply(segSpace);
		segSpace = localSpace;
		
		localD.set(0.f, 0.f, radius * m_lengthPortions[i]);
		localD = segSpace.transform(localD);
		
		segP += localD;
	}
}

Matrix33F MlRachis::getSpace(short idx) const
{
	return m_spaces[idx];
}