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
class Ray
{
public:
	Ray();
	Ray(const Vector3F& pfrom, const Vector3F& vdir, float min, float max);
	Ray(const Vector3F& pfrom, const Vector3F& pto);
	Vector3F travel(float & t) const;
	Vector3F m_origin, m_dir;
	float m_tmin, m_tmax;
};
