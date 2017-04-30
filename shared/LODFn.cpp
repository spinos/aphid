/*
 *  LODFn.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/3/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "LODFn.h"

LODFn::LODFn() 
{
	setFieldOfView(35.f);
	m_overall = 1.f;
}

void LODFn::setEyePosition(const Vector3F & pos) 
{
	m_eye = pos;
}

void LODFn::setFieldOfView(const float & fov) 
{
	m_fov = tan(fov * .5f * 3.14f / 180.f);
}

float LODFn::computeLOD(const Vector3F & p, const float r, const unsigned npix) const
{
	const Vector3F toeye = m_eye - p;

	float depth = toeye.length() - r;
	if(depth < 0.f) return m_overall;

	float w = depth * m_fov;
	
	float np = r / w * 2000;
	if(np > npix) return m_overall;
	return 0.01f + (0.99f * np / npix) * m_overall;
}

void LODFn::setOverall(float x)
{
	m_overall = x;
	if(m_overall > 1.f) m_overall = 1.f;
	else if(m_overall < 0.01f) m_overall = 0.01f;
}

float LODFn::overall() const
{
	return m_overall;
}
