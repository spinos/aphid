/*
 *  PerspectiveCamera.cpp
 *  fit
 *
 *  Created by jian zhang on 4/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include "PerspectiveCamera.h"
#include <cmath>

namespace aphid {

PerspectiveCamera::PerspectiveCamera()
{
	m_fov = 35.f;
	m_2tanfov = tan(m_fov / 360.f * 3.1415927f) * 2.f; // half fov angle * 2
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

/// width of near clipping plane
float PerspectiveCamera::frameWidth() const
{
	return m_2tanfov * m_nearClipPlane;
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
	float cx, cy;
	getScreenCoord(cx, cy, x, y);
	worldVec.x = cx * frameWidth();
	worldVec.y = cy * frameHeight();
	worldVec.z = -m_nearClipPlane;
	
	origin = fSpace.transform(worldVec);
	worldVec.normalize();
	worldVec = fSpace.transformAsNormal(worldVec);
}

void PerspectiveCamera::setFieldOfView(float x)
{
	m_fov = x;
	m_2tanfov = tan(m_fov / 360.f * 3.1415927f) * 2.f; // half fov angle * 2
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