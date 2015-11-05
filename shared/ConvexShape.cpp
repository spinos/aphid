/*
 *  Frustum.cpp
 *  
 *
 *  Created by jian zhang on 11/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ConvexShape.h"
#include <cmath>

Frustum::Frustum() {}

void Frustum::set(float nearClip, float farClip,
			float horizontalAperture, float verticalAperture,
			float angleOfView,
			const Matrix44F & space)
{
	const float frm = tan(angleOfView/360.f * 3.1415927f); // half angle
	float h_fov = horizontalAperture * frm;
    float v_fov = verticalAperture * frm;

    float fright = farClip * h_fov;
    float ftop = farClip * v_fov;

    float nright = nearClip * h_fov;
    float ntop = nearClip * v_fov;
	
	m_corners[0].set(fright, ftop, -farClip);
	m_corners[1].set(-fright, ftop, -farClip);
	m_corners[2].set(-fright, -ftop, -farClip);
	m_corners[3].set(fright, -ftop, -farClip);
	m_corners[4].set(nright, ntop, -nearClip);
	m_corners[5].set(-nright, ntop, -nearClip);
	m_corners[6].set(-nright, -ntop, -nearClip);
	m_corners[7].set(nright, -ntop, -nearClip);
	
	int i = 0;
    for(; i<8; i++) m_corners[i] = space.transform(m_corners[i]);
}

Vector3F * Frustum::x()
{ return m_corners; }

Vector3F Frustum::X(int idx) const
{ return m_corners[idx]; }

int Frustum::numPoints() const
{ return 8; }

Vector3F Frustum::supportPoint(const Vector3F & v, Vector3F * localP) const
{ 
    float maxdotv = -1e8f;
    float dotv;
	
    Vector3F res, q;
    for(int i=0; i < numPoints(); i++) {
        q = m_corners[i];
        dotv = q.dot(v);
        if(dotv > maxdotv) {
            maxdotv = dotv;
            res = q;
            if(localP) *localP = q;
        }
    }
    
    return res;
}

void Frustum::split(Frustum & child0, Frustum & child1, bool alongX) const
{
	child0 = *this;
	child1 = *this;
	Vector3F * lft = child0.x();
	Vector3F * rgt = child1.x();
	Vector3F p0, p1, p2, p3;
	if(alongX) {
		p0 = (X(0) + X(1)) * .5f;
		p1 = (X(2) + X(3)) * .5f;
		p2 = (X(4) + X(5)) * .5f;
		p3 = (X(6) + X(7)) * .5f;
		lft[0] = p0;
		rgt[1] = p0;
		lft[3] = p1;
		rgt[2] = p1;
		lft[4] = p2;
		rgt[5] = p2;
		lft[7] = p3;
		rgt[6] = p3;
	}
	else {
		p0 = (X(0) + X(3)) * .5f;
		p1 = (X(1) + X(2)) * .5f;
		p2 = (X(4) + X(7)) * .5f;
		p3 = (X(5) + X(6)) * .5f;
		lft[0] = p0;
		rgt[3] = p0;
		lft[1] = p1;
		rgt[2] = p1;
		lft[4] = p2;
		rgt[7] = p2;
		lft[5] = p3;
		rgt[6] = p3;
	}
}
//;~