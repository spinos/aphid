/*
 *  Frustum.cpp
 *  
 *
 *  Created by jian zhang on 11/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ConvexShape.h"
#include <Quantization.h>
#include <cmath>
namespace aphid {
    
namespace cvx {

Frustum::Frustum() {}

void Frustum::set(float nearClip, float farClip,
			float horizontalAperture, float verticalAperture,
			float angleOfView,
			const Matrix44F & space)
{
	const float frm = tan(angleOfView/360.f * 3.1415927f); // half angle
	float h_fov = horizontalAperture * frm;
    float v_fov = verticalAperture * frm;

    float fright = farClip * h_fov;
    float ftop = farClip * v_fov;

    float nright = nearClip * h_fov;
    float ntop = nearClip * v_fov;
	
	m_corners[0].set(-fright, -ftop, -farClip);
	m_corners[1].set( fright, -ftop, -farClip);
	m_corners[2].set(-fright,  ftop, -farClip);
	m_corners[3].set( fright,  ftop, -farClip);
	m_corners[4].set(-nright, -ntop, -nearClip);
	m_corners[5].set( nright, -ntop, -nearClip);
	m_corners[6].set(-nright,  ntop, -nearClip);
	m_corners[7].set( nright,  ntop, -nearClip);
	
	int i = 0;
    for(; i<8; i++) m_corners[i] = space.transform(m_corners[i]);
}

Vector3F * Frustum::x()
{ return m_corners; }

Vector3F Frustum::X(int idx) const
{ return m_corners[idx]; }

int Frustum::numPoints() const
{ return 8; }

Vector3F Frustum::supportPoint(const Vector3F & v, Vector3F * localP) const
{ 
    float maxdotv = -1e8f;
    float dotv;
	
    Vector3F res, q;
    for(int i=0; i < numPoints(); i++) {
        q = m_corners[i];
        dotv = q.dot(v);
        if(dotv > maxdotv) {
            maxdotv = dotv;
            res = q;
            if(localP) *localP = q;
        }
    }
    
    return res;
}

void Frustum::split(Frustum & child0, Frustum & child1, float alpha, bool alongX) const
{
	const float oneMAlpha = 1.f - alpha;
	child0 = *this;
	child1 = *this;
	Vector3F * lft = child0.x();
	Vector3F * rgt = child1.x();
	Vector3F p0, p1, p2, p3;
	if(alongX) {
		p0 = X(0) * oneMAlpha + X(1) * alpha;
		p1 = X(2) * oneMAlpha + X(3) * alpha;
		p2 = X(4) * oneMAlpha + X(5) * alpha;
		p3 = X(6) * oneMAlpha + X(7) * alpha;
		lft[1] = p0;
		rgt[0] = p0;
		lft[3] = p1;
		rgt[2] = p1;
		lft[5] = p2;
		rgt[4] = p2;
		lft[7] = p3;
		rgt[6] = p3;
	}
	else {
		p0 = X(0) * oneMAlpha + X(2) * alpha;
		p1 = X(1) * oneMAlpha + X(3) * alpha;
		p2 = X(4) * oneMAlpha + X(6) * alpha;
		p3 = X(5) * oneMAlpha + X(7) * alpha;
		lft[2] = p0;
		rgt[0] = p0;
		lft[3] = p1;
		rgt[1] = p1;
		lft[6] = p2;
		rgt[4] = p2;
		lft[7] = p3;
		rgt[5] = p3;
	}
}

Sphere::Sphere() {}

void Sphere::set(const Vector3F & x, const float & r)
{ m_p = x; m_r = r; }

BoundingBox Sphere::calculateBBox() const
{ return BoundingBox(m_p.x - m_r, m_p.y - m_r, m_p.z - m_r,
                    m_p.x + m_r, m_p.y + m_r, m_p.z + m_r); }

ShapeType Sphere::ShapeTypeId = TSphere;

std::string Sphere::GetTypeStr()
{ return "sphere"; }

Cube::Cube() {}

void Cube::set(const Vector3F & x, const float & r)
{ m_p = x; m_r = r; }
 
BoundingBox Cube::calculateBBox() const
{ return BoundingBox(m_p.x - m_r, m_p.y - m_r, m_p.z - m_r,
                    m_p.x + m_r, m_p.y + m_r, m_p.z + m_r); }

bool Cube::intersect(const Ray &ray, float *hitt0, float *hitt1) const
{ return calculateBBox().intersect(ray, hitt0, hitt1); }

ShapeType Cube::ShapeTypeId = TCube;

std::string Cube::GetTypeStr()
{ return "cube"; }

Capsule::Capsule() {}

void Capsule::set(const Vector3F & x0, const float & r0,
            const Vector3F & x1, const float & r1)
{
    m_p0 = x0; m_r0 = r0;
    m_p1 = x1; m_r1 = r1;
}
    
BoundingBox Capsule::calculateBBox() const
{
    BoundingBox b;
    b.expandBy(m_p0, m_r0);
    b.expandBy(m_p1, m_r1);
    return b;
}
    
ShapeType Capsule::ShapeTypeId = TCapsule;

Triangle::Triangle()
{}

void Triangle::setP(const Vector3F & p, const int & idx)
{ 
	if(idx == 0) m_p0 = p;
	else if(idx == 1) m_p1 = p;
	else m_p2 = p;
}

void Triangle::resetNC()
{ m_nc0 = m_nc1 = m_nc2 = 0; }

void Triangle::setN(const Vector3F & n, const int & idx)
{
	if(idx == 0) colnor30::encodeN(m_nc0, n);
	else if(idx == 1) colnor30::encodeN(m_nc1, n);
	else  colnor30::encodeN(m_nc2, n);
}

void Triangle::setC(const Vector3F & c, const int & idx)
{
	if(idx == 0) colnor30::encodeC(m_nc0, c);
	else if(idx == 1) colnor30::encodeC(m_nc1, c);
	else  colnor30::encodeC(m_nc2, c);
}

void Triangle::setInd(const int & x, const int & idx)
{ 
	if(idx == 0) m_nc0 = x;
	else if(idx == 1) m_nc1 = x;
	m_nc2 = x;
}

const Vector3F * Triangle::p(int idx) const
{ 
	if(idx == 0) return &m_p0;
	else if(idx == 1) return &m_p1;
	return &m_p2;
}

const Vector3F & Triangle::P(int idx) const
{ 
	if(idx == 0) return m_p0;
	else if(idx == 1) return m_p1;
	return m_p2;
}

Vector3F Triangle::N(int idx) const
{
	Vector3F r;
	if(idx == 0) colnor30::decodeN(r, m_nc0);
	else if(idx == 1) colnor30::decodeN(r, m_nc1);
	else colnor30::decodeN(r, m_nc2);
	return r;
}

const int & Triangle::ind0() const
{ return m_nc0; }

const int & Triangle::ind1() const
{ return m_nc1; }

BoundingBox Triangle::calculateBBox() const
{
	BoundingBox b;
    b.expandBy(m_p0);
    b.expandBy(m_p1);
	b.expandBy(m_p2);
    return b;
}

Vector3F Triangle::calculateNormal() const
{
	Vector3F ab = m_p1 - m_p0;
	Vector3F ac = m_p2 - m_p0;
	Vector3F nor = ab.cross(ac);
	nor.normalize();
	return nor;
}

void Triangle::translate(const Vector3F & v)
{
	m_p0 += v;
	m_p1 += v;
	m_p2 += v;
}

bool Triangle::intersect(const Ray &ray, float *hitt0, float *hitt1) const
{
	const Vector3F nor = calculateNormal();
	
	float ddotn = ray.m_dir.dot(nor);
	
	//if(!ctx->twoSided && ddotn > 0.f) return 0;
	
	float t = (m_p0.dot(nor) - ray.m_origin.dot(nor)) / ddotn;
	
	if(t < 0.f || t > ray.m_tmax) return 0;
	
	Vector3F onplane = ray.m_origin + ray.m_dir * t;
	Vector3F e01 = m_p1 - m_p0;
	Vector3F x0 = onplane - m_p0;
	if(e01.cross(x0).dot(nor) < 0.f) return 0;
	
	//printf("pass a\n");

	Vector3F e12 = m_p2 - m_p1;
	Vector3F x1 = onplane - m_p1;
	if(e12.cross(x1).dot(nor) < 0.f) return 0;
	
	//printf("pass b\n");
	
	Vector3F e20 = m_p0 - m_p2;
	Vector3F x2 = onplane - m_p2;
	if(e20.cross(x2).dot(nor) < 0.f) return 0;
	
	//printf("pass c\n");
	*hitt0 = t;
	
    return true;
}

const Vector3F & Triangle::X(int idx) const
{ return P(idx); }

const Vector3F & Triangle::supportPoint(const Vector3F & v, Vector3F * localP) const
{
	float maxdotv = -1e18f;
    float dotv;
	int ir = 0;
	
    for(int i=0; i < 3; ++i) {
        dotv = P(i).dot(v);
        if(dotv > maxdotv) {
            maxdotv = dotv;
            ir = i;
        }
    }
    if(localP) *localP = P(ir);
    return P(ir);
}

ShapeType Triangle::ShapeTypeId = TTriangle;

std::string Triangle::GetTypeStr()
{ return "triangle"; }

}

}
//;~