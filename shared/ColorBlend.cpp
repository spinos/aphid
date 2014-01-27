/*
 *  ColorBlend.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/26/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "ColorBlend.h"

ColorBlend::ColorBlend() { m_dropoff= 0.f; m_strength = 1.f; }
ColorBlend::~ColorBlend() {}

void ColorBlend::setCenter(const Vector3F & center) { m_center = center; }
void ColorBlend::setMaxDistance(const float & x) { m_maxDistance = x; }
void ColorBlend::setDropoff(const float & x) { m_dropoff = x; m_minDistance = m_maxDistance * (1.f - m_dropoff); }
void ColorBlend::setStrength(const float & x) { m_strength = x; }

void ColorBlend::blend(const Vector3F & p, const Float3 & src, Float3 * dst) const
{
	const float distance = m_center.distanceTo(p);
	if(distance >= m_maxDistance) return;
	float weight = 1.f;
	if(distance > m_minDistance) {
		weight = 1.f - (distance - m_minDistance) / (m_maxDistance - m_minDistance);
		weight *= weight;
	}
	
	weight *= m_strength;
	
	dst->x = src.x * weight + dst->x * (1.f - weight);
	dst->y = src.y * weight + dst->y * (1.f - weight);
	dst->z = src.z * weight + dst->z * (1.f - weight);
}