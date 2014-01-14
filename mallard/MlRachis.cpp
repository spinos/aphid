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
	m_angles = new Float2[x];
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
		m_angles[i].x = m_angles[i].y = 0.f;
	}
	m_bendDirection = 0.f;
}

void MlRachis::bend()
{
	for(unsigned i = 0; i < m_numSpace; i++) {
		m_spaces[i].setIdentity();
		m_spaces[i].rotateZ(m_angles[i].y);
		m_spaces[i].rotateY(m_angles[i].x);
	}
}

void MlRachis::bend(unsigned faceIdx, float patchU, float patchV, const Vector3F & oriP, const Matrix33F & space, float radius, CollisionRegion * collide)
{
	reset();
	
	Matrix33F invSpace, segSpace = space;
	Vector3F pop, topop, segU, smoothU, testP, toTestP, segP = oriP;
	Vector2F rotAngles;
	float segRot, pushAngle, curAngle, smoothAngle, segL, dL, bendWei;
	
	invSpace = segSpace;
	invSpace.inverse();
	
	const Vector3F rootUp = segSpace.transform(Vector3F::XAxis);
	const Vector3F rootFront = segSpace.transform(Vector3F::ZAxis);

	pushAngle = 0.f;
	
	testP = segP;
	segL = radius * m_lengthPortions[0];
	moveForward(segSpace, segL, testP);
	segU = collide->getClosestNormal(testP, 1000.f, pop);

	topop = pop - segP;
	pushAngle = pushToSurface(topop, invSpace);

	collide->interpolateVertexVector(faceIdx, patchU, patchV, &smoothU);
	
	Float3 rota = matchNormal(smoothU, invSpace);
	smoothAngle = rota.x;
	
	curAngle = pushAngle + smoothAngle;

	m_spaces[0].rotateZ(rota.y);
	m_spaces[0].rotateY(curAngle);
	m_angles[0].x = curAngle;
	m_angles[0].y = rota.y;
	
	rotateForward(m_spaces[0], segSpace);
	moveForward(segSpace, segL, segP);
	
	for(unsigned i = 1; i < m_numSpace; i++) {
		invSpace = segSpace;
		invSpace.inverse();
		
		testP = segP;
		segL = radius * m_lengthPortions[i];
		moveForward(segSpace, segL, testP);
		segU = collide->getClosestNormal(testP, 1000.f, pop);
		
		dL = Vector3F(testP, pop).length();
	
		if(dL < segL) bendWei = 1.f;
		else bendWei = (dL - segL)/segL/10.f;

		toTestP = testP - segP;
		topop = pop - segP;
		if(toTestP.dot(topop) > 0.f) pushAngle = pushToSurface(topop, invSpace);
		else bendWei = 0.f;
		
		collide->interpolateVertexVector(&segU);
		
		rota = matchNormal(segU, invSpace);
		
		segRot = rota.x;
		if(segRot > 0.f) segRot *= 0.5f;
		
		curAngle = pushAngle * bendWei;
		curAngle += segRot * bendWei;
		curAngle += 0.17f * (1.f - m_lengths[i] * 0.4f);
		curAngle += smoothAngle * m_lengthPortions[i] * 0.5f;
		
		m_spaces[i].rotateZ(rota.y * bendWei);
		m_spaces[i].rotateY(curAngle);
		m_angles[i].x = curAngle;
		m_angles[i].y = rota.y * bendWei;

		rotateForward(m_spaces[i], segSpace);
		moveForward(segSpace, segL, segP);
	}
	
	m_bendDirection = -rootUp.angleBetween(segU, rootFront);
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

Float3 MlRachis::matchNormal(const Vector3F & wv, const Matrix33F & space)
{
	Vector3F ov = space.transform(wv);
	ov.normalize();
	
	Vector3F va = ov;
	va.y = 0.f;
	va.z -= 0.1f;
	va.normalize();
	float a = acos(Vector3F::XAxis.dot(va));
	//if(a > .5f) a = .5f;
	if(va.z > 0.f) a = -a;
	
	Vector3F vb = ov;
	vb.z = 0.f;
	vb.normalize();
	float b = acos(Vector3F::XAxis.dot(vb));
	//if(b > .25f) b = .25f;
	if(vb.y < 0.f) b = -b;
	
	return Float3(a, b, 0.f);
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

Float2 * MlRachis::angles() const
{
	return m_angles;
}

float MlRachis::bendDirection() const
{
	return m_bendDirection;
}
