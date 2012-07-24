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
}

BaseCamera::~BaseCamera() {}

void BaseCamera::getMatrix(float* m) const
{
	Matrix44F minv = fSpace;
	minv.inverse();
	minv.transposed(m);
}
	
void BaseCamera::tumble(int x, int y)
{
	Vector3F side  = fSpace.getSide();
	Vector3F up    = fSpace.getUp();
	Vector3F front = fSpace.getFront();
	Vector3F eye = fSpace.getTranslation();	
	Vector3F view = eye - fCenterOfInterest;
	float dist = view.length();
	
	eye -= side * (float)x/200.f * dist;
	eye += up * (float)y/200.f * dist;
	
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
}

void BaseCamera::track(int x, int y)
{
	Vector3F side  = fSpace.getSide();
	Vector3F up    = fSpace.getUp();
	Vector3F eye = fSpace.getTranslation();
	Vector3F view = eye - fCenterOfInterest;
	float dist = view.length();
	eye -= side * (float)x/800.f * dist;
	eye += up * (float)y/800.f * dist;	
	
	fCenterOfInterest -= side * (float)x/800.f * dist;
	fCenterOfInterest += up * (float)y/800.f * dist;
	
	fSpace.setTranslation(eye);	
}

void BaseCamera::zoom(int y)
{
	Vector3F front = fSpace.getFront();
	Vector3F eye = fSpace.getTranslation();
	Vector3F view = eye - fCenterOfInterest;
	float dist = view.length();
	
	eye += front * (float)y/200.f * dist;
	
	fSpace.setTranslation(eye);
}
