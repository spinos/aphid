/*
 *  BaseCamera.h
 *  lbm3d
 *
 *  Created by jian zhang on 7/25/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <Matrix44F.h>
#include <Vector3F.h>
class BaseCamera {
public:
	BaseCamera();
	virtual ~BaseCamera();
	
	virtual bool isOrthographic() const;
	void reset();
	void lookFromTo(Vector3F & from, Vector3F & to);
	void setPortWidth(unsigned w);
	void setPortHeight(unsigned h);
	void setHorizontalAperture(float w);
	void updateInverseSpace();
	void getMatrix(float* m) const;
	void tumble(int x, int y);
	void track(int x, int y);
	virtual void zoom(int y);
	void moveForward(int y);
	
	char screenToWorldPoint(int x, int y, Vector3F & worldPos) const;
	void screenToWorldVector(int x, int y, Vector3F & worldVec) const;
	virtual void incidentRay(int x, int y, Vector3F & origin, Vector3F & worldVec) const;
	Vector3F eyePosition() const;
	float aspectRatio() const;
	virtual float fieldOfView() const;
	virtual float frameWidth() const;
	virtual float frameHeight() const;
	virtual float frameWidthRel() const;
	float getHorizontalAperture() const;
	virtual void frameCorners(Vector3F & bottomLeft, Vector3F & bottomRight, Vector3F & topRight, Vector3F & topLeft) const;
	
	Matrix44F fSpace, fInverseSpace;
	Vector3F fCenterOfInterest;
	unsigned fPortWidth, fPortHeight;
	float fHorizontalAperture;
};