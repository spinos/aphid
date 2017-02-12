/*
 *  BarycentricCoordinate.h
 *  hc
 *
 *  Created by jian zhang on 4/8/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_BARYCENTRIC_COORDINATE_H
#define APH_BARYCENTRIC_COORDINATE_H

#include <math/Vector3F.h>

namespace aphid {

class BarycentricCoordinate {
public:
	BarycentricCoordinate();
	void create(const Vector3F& p0, const Vector3F& p1, const Vector3F& p2);
	float project(const Vector3F & pos);
	void compute();
	void computeClosest();
	const float * getValue() const;
	
	Vector3F getP(unsigned idx) const;
	float getV(unsigned idx) const;
	
	const bool & insideTriangle() const;
	Vector3F getClosest() const;
	Vector3F getOnPlane() const;
	Vector3F getNormal() const;
	
private:
	void computeContribute(const Vector3F & q);
	void computeInsideTriangle();

	Vector3F m_p[3];
	Vector3F m_onEdge[3];
	Vector3F m_n;
	Vector3F m_closest;
	Vector3F m_onplane;
	float m_area;
/// contributes
	float m_v[3];
	bool m_isInsideTriangle;
};

}
#endif