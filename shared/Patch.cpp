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
	Vector3F n, dv, v, dp, pop;
	ctx->m_plane.getNormal(n);
	float l, ang;
	for(int i = 0; i < 4; i++) {
		v = vertex(i);
		dv = v - ctx->m_origin;
		l = dv.length();
		if(l < 10e-4) continue;
		if(dv.dot(ctx->m_front) < ctx->m_frontFacingThreshold) continue;
		
		ctx->m_plane.projectPoint(v, pop);
		
		dp = pop - ctx->m_origin;
		dp.normalize();
		
		ang = dp.angleBetween(dv, n);
		
		ctx->m_currentAngle = ang;
		pushed = true;
	}
	return pushed;
}

