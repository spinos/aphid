/*
 *  BarycentricCoordinate.cpp
 *  hc
 *
 *  Created by jian zhang on 4/8/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BarycentricCoordinate.h"

inline float barycentric_coord(float ax, float ay, float bx, float by, float x, float y)
{
	return (ay - by)*x + (bx - ax)*y +ax*by - bx*ay;
}

BarycentricCoordinate::BarycentricCoordinate() {}
void BarycentricCoordinate::create(const Vector3F& p0, const Vector3F& p1, const Vector3F& p2)
{
	m_p[0] = p0;
	m_p[1] = p1;
	m_p[2] = p2;
	
	Vector3F V[3];
	
	V[0] = p1 - p0;
	V[1] = p2 - p1;
	V[2] = p0 - p2;
	
	V[0].normalize();
	V[1].normalize();
	V[2].normalize();
	
	int a = 0, b = 1;
	
	Vector3F side = V[a];
	Vector3F front = side.cross(V[b]); front.normalize();
	Vector3F up = front.cross(side);
	
	m_space.setIdentity();
	m_space.setOrientations(side, up, front);
	m_space.setTranslation(m_p[a]);
	m_space.inverse();
	
	m_p[0] = m_space.transform(m_p[0]);
	m_p[1] = m_space.transform(m_p[1]);
	m_p[2] = m_space.transform(m_p[2]);
	
	f120 = barycentric_coord(m_p[1].x, m_p[1].y, m_p[2].x, m_p[2].y, m_p[0].x, m_p[0].y);
	f201 = barycentric_coord(m_p[2].x, m_p[2].y, m_p[0].x, m_p[0].y, m_p[1].x, m_p[1].y);
	f012 = barycentric_coord(m_p[0].x, m_p[0].y, m_p[1].x, m_p[1].y, m_p[2].x, m_p[2].y);
}
	
void BarycentricCoordinate::compute(const Vector3F & pos)
{
	Vector3F po = m_space.transform(pos);
	m_v[0] = barycentric_coord(m_p[1].x, m_p[1].y, m_p[2].x, m_p[2].y, po.x, po.y)/f120;
	m_v[1] = barycentric_coord(m_p[2].x, m_p[2].y, m_p[0].x, m_p[0].y, po.x, po.y)/f201;
	m_v[2] = barycentric_coord(m_p[0].x, m_p[0].y, m_p[1].x, m_p[1].y, po.x, po.y)/f012;			
}

const float * BarycentricCoordinate::getValue() const
{
	return m_v;
}

