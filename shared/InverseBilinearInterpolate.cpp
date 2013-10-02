/*
 *  InverseBilinearInterpolate.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 *    c ----- d
 *    |       |
 *  F |       |
 *    a ----- b
 *        E
 */

#include "InverseBilinearInterpolate.h"

InverseBilinearInterpolate::InverseBilinearInterpolate() {}
InverseBilinearInterpolate::~InverseBilinearInterpolate() {}

void InverseBilinearInterpolate::setVertices(const Vector3F & a, const Vector3F & b, const Vector3F & c, const Vector3F & d)
{
	Vector3F side = b - a; side.normalize();
	Vector3F up = c - a;
	Vector3F front = side.cross(up); front.normalize();
	up = front.cross(side); up.normalize();
	
	m_space.setIdentity();
	m_space.setTranslation(a);
	m_space.setOrientations(side, up, front);
	m_space.inverse();
	
	Vector3F A = m_space.transform(a);

	Vector3F B = m_space.transform(b);
	m_E.set(B.x, B.y);

	Vector3F C = m_space.transform(c);
	m_F.set(C.x, C.y);
	
	Vector3F D = m_space.transform(d);
	
	m_G.set(-B.x - C.x + D.x, -B.y - C.y + D.y);
}

Vector2F InverseBilinearInterpolate::evalBiLinear(const Vector2F& uv) const
{
	return m_E * uv.x + m_F * uv.y + m_G * uv.x * uv.y;
}

Vector2F InverseBilinearInterpolate::operator()(const Vector3F &P)
{
	const Vector3F q = m_space.transform(P);
	const Vector2F q2(q.x, q.y);
	Vector2F uv(0.5, 0.5);

	uv -= solve(m_E + m_G * uv.y, m_F + m_G * uv.x, evalBiLinear(uv) - q2, true);
	uv -= solve(m_E + m_G * uv.y, m_F + m_G * uv.x, evalBiLinear(uv) - q2, false);
	
	return uv;
}

Vector2F InverseBilinearInterpolate::solve(Vector2F M1, Vector2F M2, Vector2F b, bool safeInvert)
{
	float det = M1.cross(M2);
	if(safeInvert || det != 0.f) det = 1.f/det;
	return Vector2F(b.cross(M2), -b.cross(M1)) * det;
}
