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
	m_speed = 1.f;
	m_radius = 1.f;
	m_active = false;
}

RotationHandle::~RotationHandle()
{}

void RotationHandle::setRadius(float x)
{ m_radius = x; }

void RotationHandle::setSpeed(float x)
{ m_speed = x; }

bool RotationHandle::begin(const Ray * r)
{
	m_center = m_space->getTranslation();
	
	float dep;
	Vector3F pol = r->closestPointOnRay(m_center, &dep );
	const float d = pol.distanceTo(m_center);
	if(d > m_radius ) {
		return false;
	}
	
	dep -= sqrt(m_radius * m_radius - d * d );
	pol = r->travel(dep);
		
	m_lastV = pol - m_center;
	m_lastV.normalize();
	
	m_invSpace = *m_space;
	m_invSpace.inverse();
	
	m_localV = m_invSpace.transformAsNormal(m_lastV);
	m_localV.normalize();
	
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
	if(d > m_radius ) {
		return;
	}
	
	dep -= sqrt(m_radius * m_radius - d * d );
	pol = r->travel(dep);
		
	Vector3F curV = pol - m_center;
	curV.normalize();
	
	Vector3F axisC = m_lastV.cross(curV);
	axisC.normalize();
	float ang;
	
	m_invSpace = *m_space;
	m_invSpace.inverse();
	
	Vector3F lastlocalV = m_invSpace.transformAsNormal(m_lastV);
	lastlocalV.normalize();
	
	Vector3F curLocalV = m_invSpace.transformAsNormal(curV);
	curLocalV.normalize();
	
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
	m_rotAxis = axis;
	m_rotAngle = ang;
}

void RotationHandle::draw(const Matrix44F * camspace) const
{
	glClear(GL_DEPTH_BUFFER_BIT);
	glEnable(GL_STENCIL_TEST);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDepthMask(GL_FALSE);
	glStencilFunc(GL_NEVER, 1, 0xFF);
	glStencilOp(GL_REPLACE, GL_KEEP, GL_KEEP);  // draw 1s on test fail (always)

/// draw stencil pattern
	glStencilMask(0xFF);
	glClear(GL_STENCIL_BUFFER_BIT);  // needs mask=0xFF
  
    float m[16];

	Matrix33F rot = m_space->rotation();
	rot.orthoNormalize();
	rot *= m_radius;
	rot.glMatrix(m);
	
	Vector3F tv = m_space->getTranslation();
	m[12] = tv.x;
	m[13] = tv.y;
	m[14] = tv.z;
	
/// enlarged sphere
	glPushMatrix();
	glMultMatrixf(m);
	glScalef(1.09f, 1.09f, 1.09f);
    
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	drawAGlyph();
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPopMatrix();
	
/// cull behind sphere
  glDepthMask(GL_TRUE);
  glStencilMask(0x00);
  // draw where stencil's value is 0
  //glStencilFunc(GL_EQUAL, 0, 0xFF);
	
  // draw only where stencil's value is 1
  glStencilFunc(GL_EQUAL, 1, 0xFF);
  
  glPushMatrix();
	glMultMatrixf(m);
	glScalef(.97f, .97f, .97f);
    
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	drawAGlyph();
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	
	glScalef(1.031f, 1.031f, 1.031f);
	
	glDepthFunc(GL_ALWAYS);
	glBegin(GL_LINES);
	glColor3f(1,0,0);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(1.f, 0.f, 0.f);
	glColor3f(0,1,0);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, 1.f, 0.f);
	glColor3f(0,0,1);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, 1.f);
	
	if(m_active) {
		glColor3f(.1f, .1f, .1f);	
		glVertex3f(0.f, 0.f, 0.f);
		glVertex3fv((const float *)&m_localV);
	}
	glEnd();
	
	glDepthFunc(GL_LEQUAL);
	
	if(m_active) {
		switch (m_snap) {
		case saX:
			glColor3f(1.f, 1.f, 0.f);
			drawXCircle();
			glColor3f(0,1,0);
			drawYCircle();
			glColor3f(0,0,1);
			drawZCircle();
			break;
		case saY:
			glColor3f(1,0,0);
			drawXCircle();
			glColor3f(1.f, 1.f, 0.f);
			drawYCircle();
			glColor3f(0,0,1);
			drawZCircle();
			break;
		case saZ:
			glColor3f(1,0,0);
			drawXCircle();
			glColor3f(0,1,0);
			drawYCircle();
			glColor3f(1.f, 1.f, 0.f);
			drawZCircle();
			break;
		default:
			glColor3f(1,0,0);
			drawXCircle();
			glColor3f(0,1,0);
			drawYCircle();
			glColor3f(0,0,1);
			drawZCircle();
			break;
		}
	} else {
		glColor3f(1,0,0);
		drawXCircle();
		glColor3f(0,1,0);
		drawYCircle();
		glColor3f(0,0,1);
		drawZCircle();
	}
	
	glPopMatrix();
/// align to view
	rot = camspace->rotation();
	rot.orthoNormalize();
	rot *= m_radius;
	rot.glMatrix(m);
	m[12] = tv.x;
	m[13] = tv.y;
	m[14] = tv.z;
	
	if(m_active) {
		if(m_snap == saNone) {
			glColor3f(1.f, 1.f, 0.f);
			drawZCircle(m);
		} else {
			glColor3f(.1f, .1f, .1f);
			drawZCircle(m);
		}
	} else {
		glColor3f(.1f, .1f, .1f);
		drawZCircle(m);
	}
	
	glDisable(GL_STENCIL_TEST);
}

void RotationHandle::getDetlaRotation(Matrix33F & mat, const float & weight) const
{
	Quaternion q(m_rotAngle * weight, m_rotAxis);
	mat.set(q);
	mat.orthoNormalize();
}

}