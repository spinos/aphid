/*
 *  Segment.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "Segment.h"
#include <AllMath.h>
Segment::Segment() {}
Segment::Segment(const Vector3F& pfrom, const Vector3F& pto) : Ray(pfrom, pto) {}

float Segment::distanceTo(const Vector3F & po, Vector3F & closestP) const
{
	Vector3F vpo(m_origin, po);
	const float dvpo = vpo.length();
	if(dvpo < EPSILON) {
		closestP = m_origin;
		return 0.f;
	}
	
	vpo.normalize();
	
	const float angleFactor = m_dir.dot(vpo);
	const float dpol = angleFactor * dvpo;
	if(dpol > 0.f && dpol < m_tmax) {
		closestP = m_origin + m_dir * dpol;
		return sqrt(dvpo * dvpo - dpol * dpol);
	}
	
	if(angleFactor < 0.f) {
		closestP = m_origin;
		return dvpo;
	}
	
	closestP = m_origin + m_dir * m_tmax;
	return Vector3F(closestP, po).length();
}