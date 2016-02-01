/*
 *  AFrustum.h
 *  
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Matrix44F.h>

class AFrustum {
	Vector3F m_v[8];
	
public:
	void set(const float & horizontalApeture,
			const float & verticalApeture,
			const float & focalLength,
			const float & clipNear,
			const float & clipFar,
			const Matrix44F & mat);
	const Vector3F * v(int idx) const;
private:
	
};