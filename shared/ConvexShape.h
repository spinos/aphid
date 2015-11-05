/*
 *  ConvexShape.h
 *  
 *
 *  Created by jian zhang on 11/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "Matrix44F.h"

class Frustum {

	/// 1-------0
	/// |       |
	/// |  far  |
	/// 2-------3
	///
	/// 5-------4
	/// |       |
	/// |  near |
	/// 6-------7
	///
	Vector3F m_corners[8];
	
public:
	Frustum();
	
	void set(float nearClip, float farClip,
			float horizontalAperture, float verticalAperture,
			float angleOfView,
			const Matrix44F & space);
			
	Vector3F * x();
	Vector3F X(int idx) const;
	int numPoints() const;
	Vector3F supportPoint(const Vector3F & v, Vector3F * localP) const;

	/// 1----p0 p0---0 
	/// |     | |    |
	/// |     | |    |
	/// 2----p1 p1---3
	///
	/// 5----p2 p2---4
	/// |     | |    |
	/// |     | |    |
	/// 6----p3 p3---7
	///
	/// 1------------0 
	/// |            |
	/// p1-----------p0
	/// p1-----------p0 
	/// |            |
	/// 2------------3
	///
	/// 5------------4
	/// |            |
	/// p3-----------p2
	/// p3-----------p2
	/// |            |
	/// 6------------7
	///
	void split(Frustum & child0, Frustum & child1, bool alongX = true) const;
	
protected:

private:

};