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
	fSpace.setIdentity();
	fSpace.setTranslation(0, 0, 100);
	fCenterOfInterest = Vector3F(0, 0, -10);
	fPortWidth = 400;
	fPortHeight = 300;
	fHorizontalAperture = 100.f;
	updateInverseSpace();
}

BaseCamera::~BaseCamera() {}

bool BaseCamera::isOrthographic() const
{
	return true;
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
	const float scaleing = fHorizontalAperture / fPortWidth;
	eye -= side * (float)x * scaleing;
	eye += up * (float)y * scaleing;	
	
	fCenterOfInterest -= side * (float)x * scaleing;
	fCenterOfInterest += up * (float)y * scaleing;
	
	fSpace.setTranslation(eye);
	updateInverseSpace();
}

void BaseCamera::zoom(int y)
{
	if(fHorizontalAperture + y > .1f) fHorizontalAperture += (float)y * 0.5f;
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

char BaseCamera::intersection(int x, int y, Vector3F & worldPos) const
{	
	worldPos = fInverseSpace.transform(worldPos);
	
	worldPos.x = ((float)x/(float)fPortWidth - 0.5f) * fHorizontalAperture;
	worldPos.y = -((float)y/(float)fPortHeight - 0.5f) * fHorizontalAperture / aspectRatio();
	
	worldPos = fSpace.transform(worldPos);
	return 1;
}

void BaseCamera::screenToWorld(int x, int y, Vector3F & worldVec) const
{
    worldVec.x = ((float)x/(float)fPortWidth) * fHorizontalAperture;
	worldVec.y = -((float)y/(float)fPortHeight)  * fHorizontalAperture / aspectRatio();
	worldVec.z = 0.f;
	
	worldVec = fSpace.transformAsNormal(worldVec);
}

void BaseCamera::incidentRay(int x, int y, Vector3F & origin, Vector3F & worldVec) const
{
    //const Vector3F eye = fSpace.getTranslation();
    //const Vector3F view = eye - fCenterOfInterest;
    //Vector3F worldPos = fCenterOfInterest - view;
    //worldPos = fInverseSpace.transform(worldPos);
    origin.x = ((float)x/(float)fPortWidth - 0.5f) * fHorizontalAperture;
	origin.y = -((float)y/(float)fPortHeight - 0.5f) * fHorizontalAperture / aspectRatio();
	origin.z = 0.f;
	origin = fSpace.transform(origin);
	worldVec = fCenterOfInterest - fSpace.getTranslation();
}

Vector3F BaseCamera::eyePosition() const
{
    return fSpace.getTranslation();
}

float BaseCamera::getHorizontalAperture() const
{
	return fHorizontalAperture;
}

float BaseCamera::aspectRatio() const
{
	return (float)fPortWidth / (float)fPortHeight;
}
