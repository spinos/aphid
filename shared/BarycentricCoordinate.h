/*
 *  BarycentricCoordinate.h
 *  hc
 *
 *  Created by jian zhang on 4/8/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Vector3F.h>
#include <Matrix44F.h>
class BarycentricCoordinate {
public:
	BarycentricCoordinate();
	void create(const Vector3F& p0, const Vector3F& p1, const Vector3F& p2);
	void compute(const Vector3F & pos);
	const float * getValue() const;
	
private:
	Vector3F m_p[3];
	Matrix44F m_space;
	float m_v[3];
	float f120, f201, f012;
};