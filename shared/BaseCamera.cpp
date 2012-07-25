/*
 *  BaseCamera.cpp
 *  lbm3d
 *
 *  Created by jian zhang on 7/25/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseCamera.h"

BaseCamera::BaseCamera() 
{
	fSpace.setIdentity();
	fSpace.setTranslation(10, 10, 100);
	fCenterOfInterest = Vector3F(10, 10, 10);
	fPortWidth = 400;
	updateInverseSpace();
}

BaseCamera::~BaseCamera() {}

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

void BaseCamera::setVerticalAperture(float h)
{
	fVerticalAperture = h;
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
	Vector3F front = fSpace.getFront();
	Vector3F eye = fSpace.getTranslation();
	Vector3F view = eye - fCenterOfInterest;
	const float dist = view.length();
	
	eye += front * (float)y * dist /fPortWidth;
	
	fSpace.setTranslation(eye);
	updateInverseSpace();
}

char BaseCamera::intersection(int x, int y, Vector3F & worldPos) const
{	
	worldPos = fInverseSpace.transform(worldPos);
	
	worldPos.x = ((float)x/(float)fPortWidth - 0.5f) * fHorizontalAperture;
	worldPos.y = -((float)y/(float)fPortHeight - 0.5f) * fVerticalAperture;
	
	worldPos = fSpace.transform(worldPos);
	return 1;
}

void BaseCamera::screenToWorld(int x, int y, Vector3F & worldVec) const
{
    worldVec.x = (float)x;
	worldVec.y = -(float)y;
	worldVec.z = 0.f;
	
	worldVec = fSpace.transformAsNormal(worldVec);
}
