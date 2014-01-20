/*
 *  PerspectiveCamera.cpp
 *  fit
 *
 *  Created by jian zhang on 4/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <AllMath.h>
#include "PerspectiveCamera.h"

PerspectiveCamera::PerspectiveCamera()
{
	m_fov = 35.f;
}

PerspectiveCamera::~PerspectiveCamera()
{

}
	
bool PerspectiveCamera::isOrthographic() const
{
	return false;
}

float PerspectiveCamera::fieldOfView() const
{
	return m_fov;
}

float PerspectiveCamera::frameWidth() const
{
	double e = tan(m_fov/360.f*3.1415927);
	return e * 1.f * 2.f;
}

float PerspectiveCamera::frameWidthRel() const
{
	Vector3F eye = fSpace.getTranslation();	
	Vector3F view = eye - fCenterOfInterest;
	return frameWidth() * view.length() / fPortWidth;
}

void PerspectiveCamera::zoom(int y)
{
	moveForward(-y/2);
}

void PerspectiveCamera::incidentRay(int x, int y, Vector3F & origin, Vector3F & worldVec) const
{
	origin.x = ((float)x/(float)fPortWidth - 0.5f) * frameWidth();
	origin.y = -((float)y/(float)fPortHeight - 0.5f) * frameHeight();
	origin.z = -1.f;
	origin = fSpace.transform(origin);
	worldVec = origin - eyePosition();
}

void PerspectiveCamera::setFieldOfView(float x)
{
	m_fov = x;
}
