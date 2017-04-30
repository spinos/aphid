/*
 *  Segment.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "Segment.h"
namespace aphid {

Segment::Segment() 
{
	m_length = 0.f;
}

Segment::Segment(const Vector3F& pfrom, const Vector3F& pto) 
{
	m_origin = pfrom;
	m_dir = pto - pfrom;
	m_length = m_dir.length();
	m_dir.normalize();
}

float Segment::distanceTo(const Vector3F & po, Vector3F & closestP) const
{
	Vector3F vpo(m_origin, po);
	const float dvpo = vpo.length();
	if(m_length < EPSILON) {
		closestP = m_origin;
		return dvpo;
	}
	
	if(dvpo < EPSILON) {
		closestP = m_origin;
		return 0.f;
	}
	
	vpo.normalize();
	
	const float angleFactor = m_dir.dot(vpo);
	const float dpol = angleFactor * dvpo;
	if(dpol > 0.f && dpol < m_length) {
		closestP = m_origin + m_dir * dpol;
		return sqrt(dvpo * dvpo - dpol * dpol);
	}
	
	if(angleFactor < 0.f) {
		closestP = m_origin;
		return dvpo;
	}
	
	closestP = m_origin + m_dir * m_length;
	return Vector3F(closestP, po).length();
}

const float & Segment::length() const
{
	return m_length;
}

}