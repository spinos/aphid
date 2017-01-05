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

#include <AllMath.h>

namespace aphid {

class Matrix44F;

class RotationHandle {

	Matrix44F * m_space;
	Vector3F m_center;
	Vector3F m_lastV;
	float m_speed;
	bool m_active;
	
public:
	RotationHandle(Matrix44F * space);
	virtual ~RotationHandle();
	
	void setSpeed(float x);
	
	bool begin(const Ray * r);
	void end();
	void rotate(const Ray * r);
	void draw(const float * camspace) const;
	
};

}
#endif
