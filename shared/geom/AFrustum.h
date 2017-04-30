/*
 *  AFrustum.h
 *  
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <math/Vector3F.h>
#include <math/BoundingBox.h>
#include "GjkIntersection.h"

namespace aphid {
    
class Matrix44F;
class BoundingBox;

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
    void toRayFrame(Vector3F * dst, const int & gridX, const int & gridY) const;
    
    const BoundingBox calculateBBox() const;
    
    bool intersectPoint(const Vector3F & p) const;
	
    template<typename T>
    bool intersect(const T * b) const;
    
private:
	
};

template<typename T>
bool AFrustum::intersect(const T * b) const
{
    const BoundingBox bb = b->calculateBBox();
	const BoundingBox ba = calculateBBox();
	if(!ba.intersect(bb) ) {
		return false;
	}
	
    return gjk::Intersect1<T, AFrustum>::Evaluate(*b, *this);
}

}