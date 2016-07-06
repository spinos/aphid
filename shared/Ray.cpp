/*
 *  Ray.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "Ray.h"
#include <iostream>
namespace aphid {

Ray::Ray() {}

Ray::Ray(const Vector3F& pfrom, const Vector3F& vdir, float tmin, float tmax)
{
	m_origin = pfrom;
	m_dir = vdir;
	m_tmin = tmin;
	m_tmax = tmax;
	if(m_tmax > 1e7f) {
		std::cout<<"\n truncate ray max to 1e7f";
		m_tmax = 1e7f;
	}
}

Ray::Ray(const Vector3F& pfrom, const Vector3F& pto) 
{
	m_origin = pfrom;
	m_dir = pto - pfrom;
	m_tmin = 0.f;
	m_tmax = m_dir.length();
	if(m_tmax > 1e7f) {
		std::cout<<"\n truncate ray max to 1e7f";
		m_tmax = 1e7f;
	}
	m_dir.normalize();
}

Vector3F Ray::travel(const float & t) const
{ return m_origin + m_dir * t; }

Vector3F Ray::destination() const
{ return m_origin + m_dir * m_tmax; }

const Vector3F Ray::closetPointOnRay(const Vector3F & p, float * t) const
{
	float tt = m_origin.dot(m_dir) - p.dot(m_dir);
	if(t) *t = tt;
	return m_origin - m_dir * tt;
}
}