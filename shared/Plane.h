/*
 *  Plane.h
 *  mallard
 *
 *  Created by jian zhang on 8/29/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Vector3F.h>
class Plane {
public:
	Plane();
	Plane(const Vector3F & nor, const Vector3F & pop);
	void projectPoint(const Vector3F & p0, Vector3F & dst) const;
	void verbose() const;
private:
	float m_a, m_b, m_c, m_d;
};