/*
 *  FloodCondition.cpp
 *  aphid
 *
 *  Created by jian zhang on 11/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "FloodCondition.h"
#include <BaseTexture.h>
FloodCondition::FloodCondition() 
{
	m_minDistance = 0.1f;
	m_density = 0;
}

void FloodCondition::setMinDistance(float d) 
{
	m_minDistance = d;
}

const float & FloodCondition::minDistance() const { return m_minDistance; }

float FloodCondition::minDistance(const unsigned & faceIdx, const float & u, const float & v) const
{
	Float3 gray;
	m_density->sample(faceIdx, u, v, (float *)&gray);
	if(gray.x > .995f) return m_minDistance;
	return m_minDistance - m_minDistance * 0.7f * (1.f - gray.x);
}

void FloodCondition::setDensityMap(BaseTexture * tex) { m_density = tex; }

void FloodCondition::increaseNumSamples(const unsigned & faceIdx, unsigned & dst) const
{
	Float3 gray;
	m_density->sample(faceIdx, (float *)&gray);
	if(gray.x > .995f) return;
	dst += dst * 9 * (1.f - gray.x);
}

void FloodCondition::reduceScale(const unsigned & faceIdx, const float & u, const float & v, float & dst) const
{
	Float3 gray;
	m_density->sample(faceIdx, u, v, (float *)&gray);
	if(gray.x > .995f) return;
	dst -= dst * .7f * (1.f - gray.x);
}
