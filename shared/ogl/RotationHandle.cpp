/*
 *  RotationHandle.cpp
 *  
 *
 *  Created by jian zhang on 1/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "RotationHandle.h"
#include "DrawCircle.h"
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
	
	Vector3F axis = curV.cross(m_lastV);
	axis.normalize();
	
	float ang = (curV - m_lastV).length() * m_speed;
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
	
	glColor3f(1.f, 1.f, 0.f);
		
	float m[16];
	memcpy(m, camspace, 64);
	m[12] = m_center.x;
	m[13] = m_center.y;
	m[14] = m_center.z;
	m[15] = 1.f;
	
	DrawCircle dc;
	dc.drawZRing(m);
	
}

}