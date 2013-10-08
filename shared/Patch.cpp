/*
 *  Patch.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "Patch.h"

Patch::Patch() {}

Patch::Patch(const Vector3F & p0, const Vector3F & p1, const Vector3F & p2, const Vector3F & p3) : Plane (p0, p1, p2, p3) 
{
	m_segs[0] = Segment(p0, p1);
	m_segs[1] = Segment(p1, p2);
	m_segs[2] = Segment(p2, p3);
	m_segs[3] = Segment(p3, p0);
}

Patch::~Patch() {}

float Patch::planarDistanceTo(const Vector3F & po, Vector3F & closestP) const
{
	float d, minD = 10e8;
	Vector3F pt;
	for(int i = 0; i < 4; i++) {
		d = m_segs[i].distanceTo(po, pt);
		if(d < minD) {
			closestP = pt;
			minD = d;
		}
	}
	return minD;
}

Vector3F Patch::vertex(int idx) const
{
	return m_segs[idx].m_origin;
}

Vector3F Patch::center() const
{
	Vector3F v, c;
	for(int i = 0; i < 4; i++) {
		v = vertex(i);
		c += v;
	}
	c /= 4.f;
	return c;
}

/*
 *   (dv)
 *   
 *   3 --> 2
 *   ^     ^
 *   |     |
 *   |     |
 *   0 --> 1   z(du)
 *
 *   y
 */

Matrix33F Patch::tangentFrame() const
{
    Matrix33F frm;
    Vector3F du = (vertex(1) - vertex(0) + vertex(2) - vertex(3)) * .5f;
    Vector3F dv = (vertex(3) - vertex(0) + vertex(2) - vertex(1)) * .5f;
    du.normalize();
    dv.normalize();
    
    Vector3F side = du.cross(dv);
    side.normalize();
    
    Vector3F up = du.cross(side);
    up.normalize();
    
    frm.fill(side, up, du);
    return frm;
}

bool Patch::pushPlane(PushPlaneContext * ctx) const
{
	bool pushed = false;
	Vector3F selfN, n, dv, v, dp, pop;
	
	getNormal(selfN);
	ctx->m_plane.getNormal(n);
	
	//if(n.dot(selfN) < 0.f) return false;
	//if(ctx->m_front.dot(selfN) > 0.f) return false;
	
	int i;
	float l, ang;
	
	bool smallEnough = true;
	Vector3F toC = center() - ctx->m_ellipseCenter + selfN * .01f;
	toC.normalize();
	for(i = 0; i < 4; i++) {
		v = vertex(i);
		dv = v - ctx->m_ellipseCenter + selfN * .01f;
		dv.normalize();
		if(toC.dot(dv) < .98f) {
			smallEnough = false;
			break;
		}
	}
	
	bool tooFar = true;
	for(i = 0; i < 4; i++) {
		v = vertex(i);
		dv = v - ctx->m_ellipseCenter;
		l = dv.length();
		
		if(l < ctx->m_maxRadius) {
			tooFar = false;
			break;
		}
	}

	if(smallEnough) ctx->m_convergent = true;
	
	if(tooFar && smallEnough) return false;
	
	ctx->m_currentAngle = 0.f;
	bool allBellow = true;
	for(i = 0; i < 4; i++) {
		v = vertex(i);
		dv = v - ctx->m_origin;
		l = dv.length();
		if(l < 10e-5) continue;
		
		ctx->m_plane.projectPoint(v, pop);
		
		dp = pop - ctx->m_origin;
		dp.normalize();
		
		ang = dp.angleBetween(dv, n);
		
		if(ang > ctx->m_maxAngle) allBellow = false;
		
		if(ctx->m_currentAngle < ang)
			ctx->m_currentAngle = ang;
		
		pushed = true;
	}
	
	if(allBellow) return false;
	return pushed;
}

