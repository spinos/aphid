/*
 *  Ray.h
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Vector3F.h>
namespace aphid {

class Ray
{
public:
	Ray();
	Ray(const Vector3F& pfrom, const Vector3F& vdir, float tmin, float tmax);
	Ray(const Vector3F& pfrom, const Vector3F& pto);
	Vector3F travel(const float & t) const;
	Vector3F destination() const;
	const Vector3F Ray::closetPointOnRay(const Vector3F & p, float * t = NULL) const;
	Vector3F m_origin;
	float m_tmin;
	Vector3F m_dir;
	float m_tmax;
};

}
