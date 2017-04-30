/*
 *  TranslationHandle.cpp
 *  
 *
 *  Created by jian zhang on 1/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "TranslationHandle.h"
#include <math/Plane.h>
#include <math/BoundingBox.h>
#include <math/miscfuncs.h>
#include <gl_heads.h>

namespace aphid {

TranslationHandle::TranslationHandle(Matrix44F * space)
{ 
	m_space = space; 
	m_speed = 1.f;
	m_radius = 1.f;
	m_active = false;
}

TranslationHandle::~TranslationHandle()
{}

void TranslationHandle::setRadius(float x)
{ m_radius = x; }

void TranslationHandle::setSpeed(float x)
{ m_speed = x; }

bool TranslationHandle::begin(const Ray * r)
{
    m_deltaV.set(0.f, 0.f, 0.f);
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
        if(pyz.length() < 1.1f * m_radius) {
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
         if(pxz.length() < 1.1f * m_radius) {
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
        if(pxy.length() < 1.1f * m_radius){
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
    
    if(m_snap == saNone) {
        if(projectPlaneLocal(m_localV, r, xy) ) {
            if(m_localV.x > 0.f &&  m_localV.x < m_radius
                && m_localV.y > 0.f &&  m_localV.y < m_radius) {
                m_snap = saXY;
            }
        }
    }
    
    if(m_snap == saNone) {
        if(projectPlaneLocal(m_localV, r, yz) ) {
            if(m_localV.y > 0.f &&  m_localV.y < m_radius
                && m_localV.z > 0.f &&  m_localV.z < m_radius) {
                m_snap = saYZ;
            }
        }
    }
    
    if(m_snap == saNone) {
        if(projectPlaneLocal(m_localV, r, xz) ) {
            if(m_localV.x > 0.f &&  m_localV.x < m_radius
                && m_localV.z > 0.f &&  m_localV.z < m_radius) {
                m_snap = saXZ;
            }
        }
    }
    
    switch (m_snap) {
    case saX:
        m_localV.y = m_localV.z = 0.f;
        break;
    case saY:
        m_localV.x = m_localV.z = 0.f;
        break;
    case saZ:
        m_localV.x = m_localV.y = 0.f;
        break;
    case saXY:
        m_localV.z = 0.f;
        break;
    case saXZ:
        m_localV.y = 0.f;
        break;
    case saYZ:
        m_localV.x = 0.f;
        break;
    default:
        break;
    }
    
	m_active = (m_snap > saNone);
	return true;
}

void TranslationHandle::end()
{	
	m_active = false; 
	m_snap == saNone;
}

void TranslationHandle::translate(const Ray * r)
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
    
    m_deltaV.set(0.f, 0.f, 0.f);

    bool stat;
    Vector3F q;
    switch (m_snap) {
    case saX:
        stat = projectLocal(q, r, xy, xz);
        q.y = q.z = 0.f;
        break;
    case saY:
        stat = projectLocal(q, r, xy, yz);
        q.x = q.z = 0.f;
        break;
    case saZ:
        stat = projectLocal(q, r, xz, yz);
        q.x = q.y = 0.f;
        break;
    case saXY:
        stat = projectPlaneLocal(q, r, xy);
        q.z = 0.f;
        break;
    case saXZ:
        stat = projectPlaneLocal(q, r, xz);
        q.y = 0.f;
        break;
    case saYZ:
        stat = projectPlaneLocal(q, r, yz);
        q.x = 0.f;
        break;
    default:
        break;
    }
    
    if(!stat) {
        return;
    }
    
    m_deltaV = q - m_localV;
    m_deltaV *= m_speed;
    
    Vector3F wdv = m_space->transformAsNormal(m_deltaV);
    
    m_space->setTranslation(pop + wdv);
}

bool TranslationHandle::projectLocal(Vector3F & q,
                const Ray * r, const Plane & p1, const Plane & p2)
{
    const float a1 = Absolute<float>(p1.normal().dot(r->m_dir) );
    const float a2 = Absolute<float>(p2.normal().dot(r->m_dir) );
    float t;
    if(a1 > a2 ) {
        if(p1.rayIntersect(*r, q, t, true) ) {
            q = m_invSpace.transform(q);
            if(q.length() < 1.1f * m_radius) {
                return true;
            }
        }
    } else {
        if(p2.rayIntersect(*r, q, t, true) ) {
            q = m_invSpace.transform(q);
            if(q.length() < 1.1f * m_radius) {
                return true;
            }
        }
    }
    return false;
}

bool TranslationHandle::projectPlaneLocal(Vector3F & q,
                const Ray * r, const Plane & p1)
{
    float t;
    if(p1.rayIntersect(*r, q, t, true) ) {
        q = m_invSpace.transform(q);
        if(Absolute<float>(q.x) < m_radius
            && Absolute<float>(q.y) < m_radius
            && Absolute<float>(q.z) < m_radius) {
            return true;
        }
    }
    
    return false;
}

void TranslationHandle::draw(const Matrix44F * camspace) const
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
	glMultMatrixf((const GLfloat*)transbuf);
    
    glScalef(m_radius, m_radius, m_radius);
    
    BoundingBox b(-.03f, -.03f, -.03f, 1.03f , 1.03f, 1.03f);
    drawSolidBoundingBox(&b);
    
    glDepthMask(GL_TRUE);
    glStencilFunc(GL_EQUAL, 1, 0xFF);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	if(m_snap == saX) {
        glColor3f(1,1,0);
    } else {
        glColor3f(1,0,0);
    }
    drawXArrow();
    if(m_snap == saY) {
        glColor3f(1,1,0);
    } else {
        glColor3f(0,1,0);
    }
    drawYArrow();
    if(m_snap == saZ) {
        glColor3f(1,1,0);
    } else {
        glColor3f(0,0,1);
    }
	drawZArrow();
    
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
    
    if(m_snap == saXY) {
        glColor3f(1,1,0);
    } else {
        glColor3f(.1f,.1f,.1f);
    }
        glBegin(GL_LINE_STRIP);
        glVertex3f(1.f, 0.f, 0.f);
        glVertex3f(1.f, 1.f, 0.f);
        glVertex3f(0.f, 1.f, 0.f);
        glEnd();
    
    if(m_snap == saYZ) {
        glColor3f(1,1,0);
    } else {
        glColor3f(.1f,.1f,.1f);
    }
        glBegin(GL_LINE_STRIP);
        glVertex3f(0.f, 1.f, 0.f);
        glVertex3f(0.f, 1.f, 1.f);
        glVertex3f(0.f, 0.f, 1.f);
        glEnd();
    
    if(m_snap == saXZ) {
        glColor3f(1,1,0);
    } else {
        glColor3f(.1f,.1f,.1f);
    }
        glBegin(GL_LINE_STRIP);
        glVertex3f(1.f, 0.f, 0.f);
        glVertex3f(1.f, 0.f, 1.f);
        glVertex3f(0.f, 0.f, 1.f);
        glEnd();
	
	glPopMatrix();
    
    glDisable(GL_STENCIL_TEST);
}

void TranslationHandle::getDeltaTranslation(Vector3F & vec, const float & weight) const
{
	Vector3F wdv = m_space->transformAsNormal(m_deltaV);
    vec = wdv * weight;
}

}