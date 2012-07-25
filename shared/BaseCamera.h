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
	
	void setPortWidth(unsigned w);
	void setPortHeight(unsigned h);
	void setHorizontalAperture(float w);
	void setVerticalAperture(float h);
	void updateInverseSpace();
	void getMatrix(float* m) const;
	void tumble(int x, int y);
	void track(int x, int y);
	void zoom(int y);
	
	char intersection(int x, int y, Vector3F & worldPos) const;
	void screenToWorld(int x, int y, Vector3F & worldVec) const;
private:
	Matrix44F fSpace, fInverseSpace;
	Vector3F fCenterOfInterest;
	unsigned fPortWidth, fPortHeight;
	float fHorizontalAperture, fVerticalAperture;
};