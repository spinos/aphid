/*
 *  RotationHandle.cpp
 *  
 *
 *  Created by jian zhang on 1/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "RotationHandle.h"
#include <math/miscfuncs.h>
#include <gl_heads.h>

namespace aphid {

RotationHandle::RotationHandle(Matrix44F * space)
{ 
	m_space = space; 
	m_center = space->getTranslation();
	m_speed = 1.f;
	m_active = false;
}

RotationHandle::~RotationHandle()
{}

void RotationHandle::setSpeed(float x)
{ m_speed = x; }

bool RotationHandle::begin(const Ray * r)
{
	float dep;
	Vector3F pol = r->closestPointOnRay(m_center, &dep );
	const float d = pol.distanceTo(m_center);
	if(d > 1.f ) {
		return false;
	}
	
	dep -= sqrt(1.f - d * d );
	pol = r->travel(dep);
		
	m_lastV = pol - m_center;
	
	m_invSpace = *m_space;
	m_invSpace.inverse();
	
	m_localV = m_invSpace.transformAsNormal(m_lastV);
	
	if(Absolute<float>(m_localV.x) < .14f ) {
		m_snap = saX;
		m_localV.x = 0.f;
	} else if(Absolute<float>(m_localV.y) < .14f ) {
		m_snap = saY;
		m_localV.y = 0.f;
	} else if(Absolute<float>(m_localV.z) < .14f ) {
		m_snap = saZ;
		m_localV.z = 0.f;
	} else {
		m_snap = saNone;
	}
	
	m_active = true;
	return true;
}

void RotationHandle::end()
{ m_active = false; }

void RotationHandle::rotate(const Ray * r)
{ 
	if(!m_active) {
		return;
	}
	
	float dep;
	Vector3F pol = r->closestPointOnRay(m_center, &dep );
	const float d = pol.distanceTo(m_center);
	if(d > 1.f ) {
		return;
	}
	
	dep -= sqrt(1.f - d * d );
	pol = r->travel(dep);
		
	Vector3F curV = pol - m_center;
	
	Vector3F axisC = m_lastV.cross(curV);
	axisC.normalize();
	float ang;
	
	m_invSpace = *m_space;
	m_invSpace.inverse();
	
	Vector3F lastlocalV = m_invSpace.transformAsNormal(m_lastV);
	
	Vector3F curLocalV = m_invSpace.transformAsNormal(curV);
	
	Vector3F axis;
	switch (m_snap) {
		case saNone:
			axis = axisC;
			ang = (curV - m_lastV).length();
			break;
		case saX:
			axis = m_space->transformAsNormal(Vector3F::XAxis);
			lastlocalV.x = 0.f;
			curLocalV.x = 0.f;
			ang = (curLocalV - lastlocalV).length();
			break;
		case saY:
			axis = m_space->transformAsNormal(Vector3F::YAxis);
			lastlocalV.y = 0.f;
			curLocalV.y = 0.f;
			ang = (curLocalV - lastlocalV).length();
			break;
		case saZ:
			axis = m_space->transformAsNormal(Vector3F::ZAxis);
			lastlocalV.z = 0.f;
			curLocalV.z = 0.f;
			ang = (curLocalV - lastlocalV).length();
			break;
		default:
			break;
	}
	
	if(m_snap != saNone) {
		if(axis.dot(axisC) < 0.f) {
			axis.reverse();
		}
	}
	
	ang *= m_speed;
	
	Quaternion q(ang, axis);
	
	Matrix33F srot = m_space->rotation();
	
	Matrix33F eft(q);
	
	srot *= eft;
	
	m_space->setRotation(srot);
	
	m_lastV = curV;
}

void RotationHandle::draw(const float * camspace) const
{
	if(!m_active) {
		return;
	}
	
	float m[16];
	m_space->glMatrix(m);
	glPushMatrix();
    glMultMatrixf(m);
	
	glColor3f(.1f, .1f, .1f);
	glBegin(GL_LINES);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3fv((const float *)&m_localV);
	glEnd();
	
	glColor3f(1.f, 1.f, 0.f);
	
	switch (m_snap) {
		case saX:
			drawXRing();
			break;
		case saY:
			drawYRing();
			break;
		case saZ:
			drawZRing();
			break;
		default:
			break;
	}
	
	glPopMatrix();
	
	if(m_snap != saNone) {
		return;
	}
	
	memcpy(m, camspace, 64);
	m[12] = m_center.x;
	m[13] = m_center.y;
	m[14] = m_center.z;
	m[15] = 1.f;
	
	drawZRing(m);
	
}

}