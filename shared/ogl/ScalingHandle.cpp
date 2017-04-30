/*
 *  ScalingHandle.cpp
 *  
 *
 *  Created by jian zhang on 1/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ScalingHandle.h"
#include <math/Plane.h>
#include <math/BoundingBox.h>
#include <math/miscfuncs.h>
#include <gl_heads.h>

namespace aphid {

ScalingHandle::ScalingHandle(Matrix44F * space)
{ 
	m_space = space; 
	m_speed = 1.f;
	m_radius = 1.f;
	m_active = false;
	m_scaleV.set(1.f, 1.f, 1.f);
}

ScalingHandle::~ScalingHandle()
{}

void ScalingHandle::setRadius(float x)
{ m_radius = x; }

void ScalingHandle::setSpeed(float x)
{ m_speed = x; }

bool ScalingHandle::begin(const Ray * r)
{
    m_deltaV.set(1.f, 1.f, 1.f);
	m_scaleV.set(1.f, 1.f, 1.f);

    const Vector3F pop = m_space->getTranslation();
    Matrix33F rot = m_space->rotation();
    rot.orthoNormalize();
    
    m_invSpace.setRotation(rot);
    m_invSpace.setTranslation(pop);
    m_invSpace.inverse();
    
    const Vector3F s = rot.row(0);
    const Vector3F u = rot.row(1);
    const Vector3F f = rot.row(2);
    
    const Plane yz(s, pop);
    const Plane xz(u, pop);
    const Plane xy(f, pop);
    
    float tyz;
    Vector3F pyz;
    bool hitYz = yz.rayIntersect(*r, pyz, tyz, true);
    float txz;
    Vector3F pxz;
    bool hitXz = xz.rayIntersect(*r, pxz, txz, true);
    float txy;
    Vector3F pxy;
    bool hitXy = xy.rayIntersect(*r, pxy, txy, true);
    
    m_snap = saNone;
    
    if(hitYz) {
        pyz = m_invSpace.transform(pyz);
        if(pyz.length() < 2.f * m_radius) {
            m_localV = pyz;
            if(pyz.y > Absolute<float>(pyz.z) * 5.f ) {
                m_snap = saY;
            }
            if(pyz.z > Absolute<float>(pyz.y) * 5.f ) {
                m_snap = saZ;
            }
        }
    }
    
    if(m_snap == saNone) {
    
    if(hitXz) {
         pxz = m_invSpace.transform(pxz);
         if(pxz.length() < 2.f * m_radius) {
            m_localV = pxz;
            if(pxz.x > Absolute<float>(pxz.z) * 5.f ) {
                m_snap = saX;
            }
            if(pxz.z > Absolute<float>(pxz.x) * 5.f ) {
                m_snap = saZ;
            }
        }
    }
    
    }
    
    if(m_snap == saNone) {
    
    if(hitXy) {
        pxy = m_invSpace.transform(pxy);
        if(pxy.length() < 2.f * m_radius){
            m_localV = pxy; 
            if(pxy.x > Absolute<float>(pxy.y) * 5.f ) {
                m_snap = saX;
            }
            if(pxy.y > Absolute<float>(pxy.x) * 5.f ) {
                m_snap = saY;
            }
        }
    }
    
    }
    
    switch (m_snap) {
    case saX:
        m_localV.y = m_localV.z = 1.f;
        break;
    case saY:
        m_localV.x = m_localV.z = 1.f;
        break;
    case saZ:
        m_localV.x = m_localV.y = 1.f;
        break;
    default:
        break;
    }
    
	m_active = (m_snap > saNone);
	return true;
}

void ScalingHandle::end()
{ 
	m_scaleV.set(1.f, 1.f, 1.f);
	m_active = false; 
	m_snap == saNone;
}

void ScalingHandle::scale(const Ray * r)
{ 
	if(!m_active) {
		return;
	}
	
    const Vector3F pop = m_space->getTranslation();
    Matrix33F rot = m_space->rotation();
    rot.orthoNormalize();
    
    m_invSpace.setRotation(rot);
    m_invSpace.setTranslation(pop);
    m_invSpace.inverse();
    
    const Vector3F s = rot.row(0);
    const Vector3F u = rot.row(1);
    const Vector3F f = rot.row(2);
    
    const Plane yz(s, pop);
    const Plane xz(u, pop);
    const Plane xy(f, pop);
    
    m_deltaV.set(1.f, 1.f, 1.f);

    bool stat;
    Vector3F q;
    switch (m_snap) {
    case saX:
        stat = projectLocal(q, r, xy, xz);
        q.y = q.z = 1.f;
		stat = (q.x / m_radius > .2f);
        break;
    case saY:
        stat = projectLocal(q, r, xy, yz);
        q.x = q.z = 1.f;
		stat = (q.y / m_radius > .2f);
        break;
    case saZ:
        stat = projectLocal(q, r, xz, yz);
        q.x = q.y = 1.f;
		stat = (q.z / m_radius > .2f);
        break;
    default:
        break;
    }
	
	if(!stat) { 
		return;
    }
	
    m_deltaV.x = q.x / m_localV.x;
    m_deltaV.y = q.y / m_localV.y;
	m_deltaV.z = q.z / m_localV.z;
	
	m_deltaV.x = 1.f + (m_deltaV.x - 1.f) * m_speed;
    m_deltaV.y = 1.f + (m_deltaV.y - 1.f) * m_speed;
    m_deltaV.z = 1.f + (m_deltaV.z - 1.f) * m_speed;
    
    m_scaleV *= m_deltaV;
    m_space->scaleBy(m_deltaV);
	
	m_localV = q;
	
}

bool ScalingHandle::projectLocal(Vector3F & q,
                const Ray * r, const Plane & p1, const Plane & p2)
{
    const float a1 = Absolute<float>(p1.normal().dot(r->m_dir) );
    const float a2 = Absolute<float>(p2.normal().dot(r->m_dir) );
    float t;
    if(a1 > a2 ) {
        if(p1.rayIntersect(*r, q, t, true) ) {
            q = m_invSpace.transform(q);
            if(q.length() < 2.f * m_radius) {
                return true;
            }
        }
    } else {
        if(p2.rayIntersect(*r, q, t, true) ) {
            q = m_invSpace.transform(q);
            if(q.length() < 2.f * m_radius) {
                return true;
            }
        }
    }
    return false;
}

void ScalingHandle::draw(const Matrix44F * camspace) const
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
    
	glPushMatrix();
	
	float transbuf[16];
	m_space->glMatrix(transbuf);
	Vector3F s = m_space->getSide();
	s.normalize();
	transbuf[0] = s.x;
	transbuf[1] = s.y;
	transbuf[2] = s.z;
	
	Vector3F u = m_space->getUp();
	u.normalize();
	transbuf[4] = u.x;
	transbuf[5] = u.y;
	transbuf[6] = u.z;
	
	Vector3F f = m_space->getFront();
	f.normalize();
	transbuf[8] = f.x;
	transbuf[9] = f.y;
	transbuf[10] = f.z;
	
	glMultMatrixf((const GLfloat*)transbuf);
    
    glScalef(m_radius, m_radius, m_radius);
    
    const BoundingBox b(-.13f, -.13f, -.13f, 2.13f , 2.13f, 2.13f);
    drawSolidBoundingBox(&b);
    
    glDepthMask(GL_TRUE);
    glStencilFunc(GL_EQUAL, 1, 0xFF);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	
	glColor3f(.1f, .1f, .1f);
	glBegin(GL_LINES);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(m_scaleV.x, 0.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, m_scaleV.y, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, m_scaleV.z);
	glEnd();
	
	if(m_snap == saX) {
        glColor3f(1,1,0);
    } else {
        glColor3f(1,0,0);
    }
    
	const BoundingBox bh(-.1f, -.1f, -.1f, .1f , .1f, .1f);
	
	glTranslatef(m_scaleV.x, 0.f, 0.f);
    drawSolidBoundingBox(&bh);
    glTranslatef(-m_scaleV.x, 0.f, 0.f);
	
    if(m_snap == saY) {
        glColor3f(1,1,0);
    } else {
        glColor3f(0,1,0);
    }
    
	glTranslatef(0.f, m_scaleV.y, 0.f);
    drawSolidBoundingBox(&bh);
    glTranslatef(0.f, -m_scaleV.y, 0.f);
	
    if(m_snap == saZ) {
        glColor3f(1,1,0);
    } else {
        glColor3f(0,0,1);
    }
	
	glTranslatef(0.f, 0.f, m_scaleV.z);
    drawSolidBoundingBox(&bh);
	
	glPopMatrix();
    
    glDisable(GL_STENCIL_TEST);
}

void ScalingHandle::getDeltaScaling(Vector3F & vec, const float & weight) const
{
	const Vector3F vone(1.f, 1.f, 1.f);
	if(!m_active) {
		vec = vone;
		return;
	}
	
	const Vector3F o2dv = m_deltaV - vone;
    vec = vone + o2dv * weight;
}

}