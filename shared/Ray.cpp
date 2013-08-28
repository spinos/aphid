/*
 *  Ray.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "Ray.h"

Ray::Ray() {}

Ray::Ray(const Vector3F& pfrom, const Vector3F& vdir, float min, float max)
{
	m_origin = pfrom;
	m_dir = vdir;
	m_tmin = min;
	m_tmax = max;
	m_plane = Plane(m_origin, m_dir);
}

Ray::Ray(const Vector3F& pfrom, const Vector3F& pto) 
{
	m_origin = pfrom;
	m_dir = pto - pfrom;
	m_tmin = 0.f;
	m_tmax = m_dir.length();
	m_dir.normalize();
	m_plane = Plane(m_origin, m_dir);
}

Vector3F Ray::travel(float & t) const
{
	return m_origin + m_dir * t;
}
