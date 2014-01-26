/*
 *  ColorBlend.h
 *  aphid
 *
 *  Created by jian zhang on 1/26/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>

class ColorBlend {
public:
	ColorBlend();
	virtual ~ColorBlend();
	
	void setCenter(const Vector3F & center);
	void setMaxDistance(const float & x);
	void setDropoff(const float & x);
	void setStrength(const float & x);
	
	void blend(const Vector3F & p, const Float3 & src, Float3 * dst) const;
private:
	Vector3F m_center;
	float m_maxDistance, m_minDistance, m_dropoff, m_strength;
};