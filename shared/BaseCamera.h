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
	
	void getMatrix(float* m) const;
	void tumble(int x, int y);
	void track(int x, int y);
	void zoom(int y);
private:
	Matrix44F fSpace;
	Vector3F fCenterOfInterest;
};