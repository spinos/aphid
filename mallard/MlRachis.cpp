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

MlRachis::MlRachis() : m_spaces(0), m_lengths(0), m_lengthPortions(0), m_angles(0) {}
MlRachis::~MlRachis() 
{
	if(m_spaces) delete[] m_spaces;
	if(m_angles) delete[] m_angles;
	if(m_lengths) delete[] m_lengths;
	if(m_lengthPortions) delete[] m_lengthPortions;
}

void MlRachis::create(unsigned x)
{
	if(m_spaces) delete[] m_spaces;
	if(m_angles) delete[] m_angles;
	if(m_lengths) delete[] m_lengths;
	if(m_lengthPortions) delete[] m_lengthPortions;
	m_numSpace = x;
	m_spaces = new Matrix33F[x];
	m_angles = new float[x];
	m_lengths = new float[x];
	m_lengthPortions = new float[x];
}

void MlRachis::computeLengths(float * segL, float fullL)
{
	float acc = 0.f;
    for(unsigned i = 0; i < m_numSpace; i++) {
		m_lengthPortions[i] = segL[i] / fullL;
		acc += m_lengthPortions[i];
		m_lengths[i] = acc * acc;
	}
}

void MlRachis::reset()
{
	for(unsigned i = 0; i < m_numSpace; i++) {
		m_spaces[i].setIdentity();
		m_angles[i] = 0.f;
	}
}

void MlRachis::bend(unsigned faceIdx, float patchU, float patchV, const Vector3F & oriP, const Matrix33F & space, float radius, CollisionRegion * collide)
{
	reset();
	
	Matrix33F invSpace, segSpace = space;
	Vector3F pop, topop, segU, smoothU, testP, segP = oriP;
	Vector2F rotAngles;
	float segRot, pushAngle, curAngle, smoothAngle;
	
	invSpace = segSpace;
	invSpace.inverse();
	
	pushAngle = 0.f;
	
	testP = segP;
	moveForward(segSpace, radius * m_lengthPortions[0], testP);
	segU = collide->getClosestNormal(testP, 1000.f, pop);

	topop = pop - segP;
	pushAngle = pushToSurface(topop, invSpace);

	collide->interpolateVertexVector(faceIdx, patchU, patchV, &smoothU);

	smoothAngle = matchNormal(smoothU, invSpace);
	
	float smoothPortion = smoothAngle;
	smoothPortion *= 0.66f;
	
	curAngle = pushAngle + smoothAngle;

	m_spaces[0].rotateY(curAngle);
	m_angles[0] = curAngle;
	
	rotateForward(m_spaces[0], segSpace);
	moveForward(segSpace, radius * m_lengthPortions[0], segP);
	
	for(unsigned i = 1; i < m_numSpace; i++) {
		invSpace = segSpace;
		invSpace.inverse();
		
		testP = segP;
		moveForward(segSpace, radius * m_lengthPortions[i], testP);
		segU = collide->getClosestNormal(testP, 1000.f, pop);

		topop = pop - segP;
		pushAngle = pushToSurface(topop, invSpace);
		segRot = matchNormal(segU, invSpace);
		
		//curAngle = pushAngle + segRot * (1.f - fullPitch * 0.5f) * (1.f - smoothPortion) + 0.15f * (1.f - m_lengths[i] * 0.5f) + smoothAngle * m_lengthPortions[i] * 0.5f;
		curAngle = pushAngle;
		curAngle += segRot;
		curAngle += 0.15f * (1.f - m_lengths[i] * 0.5f);
		curAngle += smoothAngle * m_lengthPortions[i] * 0.5f;
		m_spaces[i].rotateY(curAngle);
		m_angles[i] = curAngle;

		rotateForward(m_spaces[i], segSpace);
		moveForward(segSpace, radius * m_lengthPortions[i], segP);
	}
}

void MlRachis::curl(const float & fullPitch)
{
	for(unsigned i = 1; i < m_numSpace; i++) m_spaces[i].rotateY(fullPitch * m_lengths[i] * 0.5f);
}

Matrix33F MlRachis::getSpace(short idx) const
{
	return m_spaces[idx];
}

char MlRachis::isInside(const Vector3F & t, const Vector3F & onp, const Vector3F & nor)
{
	return Vector3F(t, onp).dot(nor) > 0.f;
}

float MlRachis::pushToSurface(const Vector3F & wv, const Matrix33F & space)
{
	Vector3F ov = space.transform(wv);
	ov.normalize();
	ov.y = 0.f;
	ov.x += 0.1f;
	ov.normalize();
	float a = acos(Vector3F::ZAxis.dot(ov));
	if(ov.x < 0.f) a = 0.f;
	return a;
}

float MlRachis::matchNormal(const Vector3F & wv, const Matrix33F & space)
{
	Vector3F ov = space.transform(wv);
	ov.normalize();
	ov.y = 0.f;
	ov.z -= 0.1f;
	ov.normalize();
	float a = acos(Vector3F::XAxis.dot(ov));
	if(ov.z > 0.f) a = -a;
	return a;
}

float MlRachis::bouncing(const Vector3F & a, const Vector3F & b, const Vector3F & c)
{
	float alpha = Vector3F(a, b).normal().dot(Vector3F(a, c).normal());
	if(alpha < 0.f) return 0.f;
	float lac = Vector3F(a, c).length() * alpha - Vector3F(a, b).length();
	float beta = asin(lac / Vector3F(a, c).length());
	return 2.f - beta - alpha;
}

float MlRachis::distanceFactor(const Vector3F & a, const Vector3F & b, const Vector3F & c)
{
	float facing = Vector3F(a, b).normal().dot(Vector3F(a, c).normal());
	float lac = Vector3F(a, c).length() * facing;
	float lab = Vector3F(a, b).length();
	if(lac > lab) return 0.f;
	float ang = 1.f - lac / lab * (lac / lab);
	return ang * 3;
}

void MlRachis::moveForward(const Matrix33F & space, float distance, Vector3F & dst)
{
	Vector3F wv = space.transform(Vector3F::ZAxis);
	wv.normalize();
	dst += wv * distance;
}

void MlRachis::rotateForward(const Matrix33F & space, Matrix33F & dst)
{
	Matrix33F s = space;
	s.multiply(dst);
	dst = s;
}
