/*
 *  RotationHandle.h
 *  
 *
 *  Created by jian zhang on 1/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_ROTATION_HANDLE
#define APH_OGL_ROTATION_HANDLE

#include <math/Matrix44F.h>
#include <math/Ray.h>
#include "DrawCircle.h"

namespace aphid {

class RotationHandle : public DrawCircle {

	Matrix44F * m_space;
	Matrix44F m_invSpace;
	Vector3F m_center;
	Vector3F m_lastV;
	Vector3F m_localV;
	float m_radius;
	float m_speed;
	bool m_active;
	
	enum SnapAxis {
		saNone = 0,
		saX = 1,
		saY = 2,
		saZ = 3
	};
	
	SnapAxis m_snap;
	
public:
	RotationHandle(Matrix44F * space);
	virtual ~RotationHandle();
	
	void setRadius(float x);
	void setSpeed(float x);
	
	bool begin(const Ray * r);
	void end();
	void rotate(const Ray * r);
	void draw(const Matrix44F * camspace) const;
	
};

}
#endif
