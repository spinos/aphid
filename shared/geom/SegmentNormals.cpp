/*
 *  SegmentNormals.cpp
 *  
 *
 *  Created by jian zhang on 7/29/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SegmentNormals.h"
#include <math/Vector3F.h>

namespace aphid {

SegmentNormals::SegmentNormals(int nseg)
{
	m_ups = new Vector3F[nseg];
}

SegmentNormals::~SegmentNormals()
{
	delete[] m_ups;
}

const Vector3F& SegmentNormals::getNormal(int i) const
{ return m_ups[i]; }

void SegmentNormals::calculateFirstNormal(const Vector3F& p0p1,
					const Vector3F& ref)
{
	Vector3F side = p0p1.cross(ref);
	side.normalize();
	m_ups[0] = side.cross(p0p1);
	m_ups[0].normalize();
}

void SegmentNormals::calculateNormal(int i, const Vector3F& p0p1,
						const Vector3F& p1p2,
						const Vector3F& p1pm02)
{
	Vector3F ref = m_ups[i-1];

	Vector3F side = p1p2.cross(ref);
	float lside = side.length();
	if(lside < 0.1f) {
/// p1p2 parallel to last_nml
		ref = p1pm02;
		if(ref.dot(m_ups[i-1]) < 0.f) {
/// rotate outside
			ref *= -1.f;
		}
		
		side = p1p2.cross(ref);
		side.normalize();
		
	} else {
		side /= lside;
	}
	
	m_ups[i] = side.cross(p1p2);
	m_ups[i].normalize();
	
}

}
