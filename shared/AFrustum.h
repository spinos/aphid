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
namespace aphid {

class AFrustum {
	Vector3F m_v[8];
	
public:
	void set(const float & hfov,
			const float & aspectRatio,
			const float & clipNear,
			const float & clipFar,
			const Matrix44F & mat);
	void setOrtho(const float & hwith,
			const float & aspectRatio,
			const float & clipNear,
			const float & clipFar,
			const Matrix44F & mat);
	const Vector3F * v(int idx) const;
	Vector3F X(int idx) const;
	Vector3F supportPoint(const Vector3F & v, Vector3F * localP = 0) const;
	Vector3F center() const;
/// origin at left-up corner, right and down deviation, 
/// at near and far clip
    void toRayFrame(Vector3F * dst) const;
    
private:
	
};

}