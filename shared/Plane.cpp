/*
 *  Plane.cpp
 *  mallard
 *
 *  Created by jian zhang on 8/29/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "Plane.h"
#include <iostream>
Plane::Plane() 
{
}

Plane::Plane(const Vector3F & nor, const Vector3F & pop)
{
	Vector3F nn = nor.normal();
	m_a = nn.x;
	m_b = nn.y;
	m_c = nn.z;
	m_d = - pop.dot(nn);
}

void Plane::projectPoint(const Vector3F & p0, Vector3F & dst) const
{
	float tt = - (p0.x * m_a + p0.y * m_b + p0.z * m_c + m_d);
	dst.x = p0.x - m_a * tt;
	dst.y = p0.y - m_b * tt;
	dst.z = p0.z - m_c * tt;
}

void Plane::verbose() const
{
	printf("ray %f %f %f %f ", m_a, m_b, m_c, m_d);
}
//:~