/*
 *  PerspectiveCamera.cpp
 *  fit
 *
 *  Created by jian zhang on 4/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include "PerspectiveCamera.h"

namespace aphid {

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

// at depth 1.0f
float PerspectiveCamera::frameWidth() const
{
	double e = tan(m_fov/360.f*3.1415927f); // half fov angle
	return e * 2.f;
}

float PerspectiveCamera::frameWidthRel() const
{
	Vector3F eye = fSpace.getTranslation();	
	Vector3F view = eye - fCenterOfInterest;
	return frameWidth() * ( view.length() / fPortWidth );
}

void PerspectiveCamera::zoom(int y)
{
	moveForward(-y/2);
}

void PerspectiveCamera::incidentRay(int x, int y, Vector3F & origin, Vector3F & worldVec) const
{
	worldVec.x = ((float)x/(float)fPortWidth - 0.5f) * frameWidth();
	worldVec.y = -((float)y/(float)fPortHeight - 0.5f) * frameHeight();
	worldVec.z = -2.f;
	worldVec = fSpace.transformAsNormal(worldVec);
	origin = fSpace.getTranslation();
}

void PerspectiveCamera::setFieldOfView(float x)
{
	m_fov = x;
}

void PerspectiveCamera::screenToWorldVectorAt(int x, int y, float depth, Vector3F & worldVec) const
{
	Vector3F vecNear;
	screenToWorldVector(x, y, vecNear);
	const Vector3F vecFar = vecNear * m_farClipPlane / m_nearClipPlane;
	float alpha = (depth - m_nearClipPlane) / (m_farClipPlane - m_nearClipPlane);
	if(alpha < 0.f) alpha = 0.f;
	else if(alpha > 1.f) alpha = 1.f;
	worldVec = vecFar * alpha + vecNear * (1.f - alpha);
}

}