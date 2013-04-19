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
	void computeClosest();
	const float * getValue() const;
	
	Vector3F getP(unsigned idx) const;
	float getV(unsigned idx) const;
	
	char insideTriangle() const;
	Vector3F getClosest() const;
	Vector3F getOnPlane() const;
	
private:
	Vector3F m_p[3];
	Vector3F m_n;
	Vector3F m_closest;
	Vector3F m_onplane;
	float m_area;
	float m_v[3];
};