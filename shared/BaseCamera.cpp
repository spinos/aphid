/*
 *  BaseCamera.cpp
 *  lbm3d
 *
 *  Created by jian zhang on 7/25/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseCamera.h"
#include <iostream>
BaseCamera::BaseCamera() 
{
    m_nearClipPlane = 1.f;
    m_farClipPlane = 1000.f;
	fPortWidth = 400;
	fPortHeight = 300;
	reset(Vector3F(0.f, 0.f, 0.f));
}

BaseCamera::~BaseCamera() {}

bool BaseCamera::isOrthographic() const
{
	return true;
}

void BaseCamera::reset(const Vector3F & pos)
{
	fHorizontalAperture = 100.f;
	
	fSpace.setIdentity();
	fSpace.setTranslation(pos + Vector3F(0.f, 0.f, 100.f));
	fCenterOfInterest = pos;
	updateInverseSpace();
}

void BaseCamera::lookFromTo(Vector3F & from, Vector3F & to)
{
	fSpace.setIdentity();
	fSpace.setTranslation(from.x, from.y, from.z);
	fCenterOfInterest = to;
	updateInverseSpace();
}

void BaseCamera::setPortWidth(unsigned w)
{
	fPortWidth = w;
}

void BaseCamera::setPortHeight(unsigned h)
{
	fPortHeight = h;
}

void BaseCamera::setHorizontalAperture(float w)
{
	fHorizontalAperture = w;
}

void BaseCamera::updateInverseSpace()
{
	fInverseSpace = fSpace;
	fInverseSpace.inverse();
}

void BaseCamera::getMatrix(float* m) const
{
	fInverseSpace.transposed(m);
}
	
void BaseCamera::tumble(int x, int y)
{
	Vector3F side  = fSpace.getSide();
	Vector3F up    = fSpace.getUp();
	Vector3F front = fSpace.getFront();
	Vector3F eye = fSpace.getTranslation();	
	Vector3F view = eye - fCenterOfInterest;
	const float dist = view.length();
	const float scaleing = dist / fPortWidth * 2.f;
	eye -= side * (float)x * scaleing;
	eye += up * (float)y * scaleing;
	
	view = eye - fCenterOfInterest;
	view.normalize();
	
	eye = fCenterOfInterest + view * dist;
	fSpace.setTranslation(eye);
	
	front = view;
	
	side = up.cross(front);
	side.y = 0.f;
	side.normalize();
	
	up = front.cross(side);
	up.normalize();
	
	fSpace.setOrientations(side, up, front);
	updateInverseSpace();
}

void BaseCamera::track(int x, int y)
{
	Vector3F side  = fSpace.getSide();
	Vector3F up    = fSpace.getUp();
	Vector3F eye = fSpace.getTranslation();
	const float scaleing = frameWidthRel();
	eye -= side * (float)x * scaleing;
	eye += up * (float)y * scaleing;	
	
	fCenterOfInterest -= side * (float)x * scaleing;
	fCenterOfInterest += up * (float)y * scaleing;
	
	fSpace.setTranslation(eye);
	updateInverseSpace();
}

void BaseCamera::zoom(int y)
{
	fHorizontalAperture *= (float)(fPortWidth + fPortHeight + y * 3)/(float)(fPortWidth + fPortHeight);
	if(fHorizontalAperture < .05f) fHorizontalAperture = .05f;
}

void BaseCamera::moveForward(int y)
{
	Vector3F front = fSpace.getFront();
	Vector3F eye = fSpace.getTranslation();
	Vector3F view = eye - fCenterOfInterest;
	const float dist = view.length();
	
	const float fra = (float)y/100.f;
	
	eye += front * dist * -fra;
	fSpace.setTranslation(eye);
	updateInverseSpace();
}

char BaseCamera::screenToWorldPoint(int x, int y, Vector3F & worldPos) const
{	
	worldPos.x = ((float)x/(float)fPortWidth - 0.5f) * frameWidth();
	worldPos.y = -((float)y/(float)fPortHeight - 0.5f) * frameHeight();
	worldPos.z = -1.f;
	
	worldPos = fSpace.transform(worldPos);
	return 1;
}

void BaseCamera::screenToWorldVector(int x, int y, Vector3F & worldVec) const
{
    worldVec.x = ((float)x/(float)fPortWidth) * frameWidth();
	worldVec.y = -((float)y/(float)fPortHeight)  * frameHeight();
	worldVec.z = 0.f;
	
	worldVec = fSpace.transformAsNormal(worldVec);
}

void BaseCamera::incidentRay(int x, int y, Vector3F & origin, Vector3F & worldVec) const
{
    origin.x = ((float)x/(float)fPortWidth - 0.5f) * frameWidth();
	origin.y = -((float)y/(float)fPortHeight - 0.5f) * frameHeight();
	origin.z = -1.f;
	origin = fSpace.transform(origin);
	worldVec = fCenterOfInterest - fSpace.getTranslation();
}

Vector3F BaseCamera::eyePosition() const
{
    return fSpace.getTranslation();
}

Vector3F BaseCamera::eyeDirection() const
{
	return Vector3F(fSpace.M(2, 0), fSpace.M(2, 1), fSpace.M(2, 2));
}

float BaseCamera::getHorizontalAperture() const
{
	return fHorizontalAperture;
}

float BaseCamera::aspectRatio() const
{
	return (float)fPortWidth / (float)fPortHeight;
}

float BaseCamera::nearClipPlane() const
{
    return m_nearClipPlane;
}

float BaseCamera::farClipPlane() const
{
    return m_farClipPlane;
}

float BaseCamera::fieldOfView() const
{
	return getHorizontalAperture();
}

float BaseCamera::frameWidth() const
{
	return getHorizontalAperture();
}

float BaseCamera::frameHeight() const
{
	return frameWidth() / aspectRatio();
}

float BaseCamera::frameWidthRel() const
{
	return frameWidth() / fPortWidth;
}

void BaseCamera::frameCorners(Vector3F & bottomLeft, Vector3F & bottomRight, Vector3F & topRight, Vector3F & topLeft) const
{
	const float halfw = frameWidth() * 0.499f;
	const float halfh = frameHeight() * 0.499f;
	bottomLeft.setZero(); bottomLeft.z = -1.01f;
	bottomLeft.x = -halfw;
	bottomLeft.y = -halfh;
	bottomLeft = fSpace.transform(bottomLeft);
	
	bottomRight.setZero(); bottomRight.z = -1.01f;
	bottomRight.x = halfw;
	bottomRight.y = -halfh;
	bottomRight = fSpace.transform(bottomRight);
	
	topRight.setZero(); topRight.z = -1.01f;
	topRight.x = halfw;
	topRight.y = halfh;
	topRight = fSpace.transform(topRight);
	
	topLeft.setZero(); topLeft.z = -1.01f;
	topLeft.x = -halfw;
	topLeft.y = halfh;
	topLeft = fSpace.transform(topLeft);
}

void BaseCamera::copyTransformFrom(BaseCamera * another)
{
	fSpace = another->fSpace;
	fInverseSpace = another->fInverseSpace;
	fCenterOfInterest = another->fCenterOfInterest;
}

void BaseCamera::setNearClipPlane(float x)
{
	m_nearClipPlane = x;
}

void BaseCamera::setFarClipPlane(float x)
{	
	m_farClipPlane = x;
}

 void BaseCamera::setFieldOfView(float) {}
//:~
