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

Plane::Plane(const Vector3F & p0, const Vector3F & p1, const Vector3F & p2, const Vector3F & p3)
{
	create(p0, p1, p2, p3);
}

Plane::~Plane() {}

void Plane::create(const Vector3F & p0, const Vector3F & p1, const Vector3F & p2, const Vector3F & p3)
{
	Vector3F cen = p0 * 0.25f + p1 * 0.25f + p2 * 0.25f + p3 * 0.25f;
	Vector3F c0 = p2 - p0;
	Vector3F c1 = p3 - p1;
	Vector3F nn = c0.cross(c1);
	nn.normalize();
	m_a = nn.x;
	m_b = nn.y;
	m_c = nn.z;
	m_d = - cen.dot(nn);
}

Vector3F Plane::normal() const
{
	return Vector3F(m_a, m_b, m_c);
}

void Plane::getNormal(Vector3F & nor) const
{
	nor.x = m_a;
	nor.y = m_b;
	nor.z = m_c;
}

void Plane::projectPoint(const Vector3F & p0, Vector3F & dst) const
{
	float tt = -(p0.x * m_a + p0.y * m_b + p0.z * m_c + m_d) / ( - m_a * m_a - m_b * m_b - m_c * m_c);
	dst.x = p0.x - m_a * tt;
	dst.y = p0.y - m_b * tt;
	dst.z = p0.z - m_c * tt;
}

bool Plane::rayIntersect(const Ray & ray, Vector3F & dst, float & t, bool twoSided) const
{
	float below = ray.m_dir.x * m_a + ray.m_dir.y * m_b + ray.m_dir.z * m_c;
	if(!twoSided) {
		if(below > -EPSILON) return false;
	}
	else {
		if(below > -EPSILON && below < EPSILON) return false;
	}
	t = - (ray.m_origin.x * m_a + ray.m_origin.y * m_b + ray.m_origin.z * m_c + m_d) / below;
	dst = ray.m_origin + ray.m_dir * t;
	return true;
}

void Plane::verbose() const
{
	printf("ray %f %f %f %f ", m_a, m_b, m_c, m_d);
}
//:~