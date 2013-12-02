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
		m_angles[i] = acc * acc;
	}
}

void MlRachis::reset()
{
	for(unsigned i = 0; i < m_numSpace; i++) m_spaces[i].setIdentity(); 
}

void MlRachis::bend(unsigned faceIdx, float patchU, float patchV, const Vector3F & oriP, const Matrix33F & space, float radius, CollisionRegion * collide, const float & fullPitch)
{
	reset();
	
	Vector3F zdir(0.f, 0.f, 1.f);
	zdir = space.transform(zdir);
	Vector3F xdir(1.f, 0.f, 0.f);
	xdir = space.transform(xdir);
	
	Matrix33F localSpace, invSpace, segSpace = space;
	Vector3F localD, localU, pop, topop, segU, smoothU, segP = oriP;
	Vector2F rotAngles;
	float segRot, pushAngle, curAngle, smoothAngle;
	
	invSpace = segSpace;
	invSpace.inverse();

	localD.set(0.f, 0.f, 1.f);
	localD = segSpace.transform(localD);
	localD.normalize();
	
	localU.set(1.f, 0.f, 0.f);
	localU = segSpace.transform(localU);
	localU.normalize();
	
	pushAngle = 0.f;
	segU = collide->getClosestNormal(segP + localD * radius * m_lengthPortions[0], radius * 2.f, pop);
	pop += segU * 0.1f * radius * m_lengthPortions[0];
	
	topop = pop - segP;
	topop = invSpace.transform(topop);
	if(topop.x > 0.f) {
		topop.y = 0.f;
		pushAngle = acos(Vector3F::ZAxis.dot(topop.normal()));
	}

	smoothAngle = 0.f;
	collide->interpolateVertexVector(faceIdx, patchU, patchV, &smoothU);
	smoothU -= oriP;
	smoothU = invSpace.transform(smoothU);

	if(smoothU.z < 0.f) {
		smoothU.y = 0.f;
		smoothAngle = acos(Vector3F::XAxis.dot(smoothU.normal()));
	}
	
	if(smoothAngle > 1.5f) smoothAngle = 1.5f;
	
	float smoothPortion = smoothAngle;
	smoothPortion *= 0.66f;
	
	curAngle = pushAngle;

	m_spaces[0].rotateY(curAngle);

	localSpace = m_spaces[0];
	localSpace.multiply(segSpace);
	segSpace = localSpace;
	
	localD.set(0.f, 0.f, radius * m_lengthPortions[0]);
	localD = segSpace.transform(localD);
	
	segP += localD;
	
	for(unsigned i = 1; i < m_numSpace; i++) {
		invSpace = segSpace;
		invSpace.inverse();

		localD.set(0.f, 0.f, 1.f);
		localD = segSpace.transform(localD);
		localD.normalize();
		
		localU.set(1.f, 0.f, 0.f);
		localU = segSpace.transform(localU);
		localU.normalize();
		
		pushAngle = 0.f;
		segU = collide->getClosestNormal(segP + localD * radius * m_lengthPortions[i], radius * 2.f, pop);

		pop += segU * 0.1f * radius * m_lengthPortions[i];
		topop = pop - segP;
		topop = invSpace.transform(topop);
		if(topop.x > 0.f) {
			topop.y = 0.f;
			pushAngle = acos(Vector3F::ZAxis.dot(topop.normal()));
		}

		segU = invSpace.transform(segU);
		segU.y = 0.f;
		
		segRot = acos(segU.normal().dot(Vector3F::XAxis));
		if(segU.z > 0.f) segRot *= -1.f;
		
		curAngle = pushAngle + segRot * (1.f - fullPitch * 0.5f) * (1.f - smoothPortion * 0.5f) + 0.15f + smoothAngle * m_lengthPortions[i] * 0.5f;
		
		m_spaces[i].rotateY(curAngle);

		localSpace = m_spaces[i];
		localSpace.multiply(segSpace);
		segSpace = localSpace;
		
		localD.set(0.f, 0.f, radius * m_lengthPortions[i]);
		localD = segSpace.transform(localD);
		
		segP += localD;
	}
	
	for(unsigned i = 1; i < m_numSpace; i++) m_spaces[i].rotateY(fullPitch * m_angles[i] * 0.5f);
}

Matrix33F MlRachis::getSpace(short idx) const
{
	return m_spaces[idx];
}