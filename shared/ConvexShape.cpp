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

Shape::Shape()
{}

Shape::~Shape()
{}

float Shape::distanceTo(const Vector3F & p) const
{ return 1e10f; }

ShapeType Shape::shapeType() const 
{ return TUnknown; }

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

ShapeType Sphere::shapeType() const 
{ return TSphere; }

float Sphere::distanceTo(const Vector3F & p) const
{ return p.distanceTo(m_p) - m_r; }

Vector3F Sphere::supportPoint(const Vector3F & v, Vector3F * localP) const
{
	Vector3F res = m_p + v.normal() * m_r;
	if(localP) *localP = res;
	return res;
}

Cube::Cube() {}

void Cube::set(const Vector3F & x, const float & r)
{ m_p = x; m_r = r; }
 
BoundingBox Cube::calculateBBox() const
{ return BoundingBox(m_p.x - m_r, m_p.y - m_r, m_p.z - m_r,
                    m_p.x + m_r, m_p.y + m_r, m_p.z + m_r); }

bool Cube::intersect(const Ray &ray, float *hitt0, float *hitt1) const
{ return calculateBBox().intersect(ray, hitt0, hitt1); }

Vector3F Cube::calculateNormal() const
{ return Vector3F::XAxis; }

ShapeType Cube::ShapeTypeId = TCube;

std::string Cube::GetTypeStr()
{ return "cube"; }

float Cube::distanceTo(const Vector3F & p) const
{
	BoundingBox bx = calculateBBox();
	float d = bx.distanceTo(p);
	if(d >= 0.f)
		return d;
		
/// inside box
	float dx = m_r - Absolute<float>(p.x - m_p.x);
	float dy = m_r - Absolute<float>(p.y - m_p.y);
	float dz = m_r - Absolute<float>(p.z - m_p.z);
	
	if(dx < dy && dx < dz)
		return -dx;
		
	if(dy < dx && dy < dz)
		return -dy;
		
	return -dz;
}

ShapeType Cube::shapeType() const 
{ return TCube; }

Box::Box() {}

void Box::set(const Vector3F & lo, const Vector3F & hi)
{ m_low = lo; m_high = hi; }

void Box::set(const float * m)
{
	m_low.set(m[0], m[1], m[2]);
	m_high.set(m[3], m[4], m[5]);
}

void Box::expand(const float & d)
{
	m_low.x -= d; m_low.y -= d; m_low.z -= d;
	m_high.x += d; m_high.y += d; m_high.z += d;
}

BoundingBox Box::calculateBBox() const
{ return BoundingBox(m_low.x, m_low.y, m_low.z,
                    m_high.x, m_high.y, m_high.z); }
					
bool Box::intersect(const Ray &ray, float *hitt0, float *hitt1) const
{ return calculateBBox().intersect(ray, hitt0, hitt1); }

Vector3F Box::calculateNormal() const
{ return Vector3F::XAxis; }

ShapeType Box::ShapeTypeId = TBox;

std::string Box::GetTypeStr()
{ return "box"; }

float Box::distanceTo(const Vector3F & p) const
{
	BoundingBox bx = calculateBBox();
	float d = bx.distanceTo(p);
	if(d >= 0.f)
		return d;
		
/// inside box
	Vector3F dp = p - bx.center();
	float dx = bx.distance(0) * .5f - Absolute<float>(dp.x);
	float dy = bx.distance(1) * .5f - Absolute<float>(dp.y);
	float dz = bx.distance(2) * .5f - Absolute<float>(dp.z);
	
	if(dx < dy && dx < dz)
		return -dx;
		
	if(dy < dx && dy < dz)
		return -dy;
		
	return -dz;
}

ShapeType Box::shapeType() const 
{ return TBox; }

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

Vector3F Triangle::C(int idx) const
{
	Vector3F r;
	if(idx == 0) colnor30::decodeC(r, m_nc0);
	else if(idx == 1) colnor30::decodeC(r, m_nc1);
	else colnor30::decodeC(r, m_nc2);
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
    b.expand(b.getLongestDistance() * 1e-4f);
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
	
	//std::cout<<"\n Triangle::intersect "<<nor<<" "<<ddotn<<" t "<<t;
	
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
	*hitt1 = t;
	
    return true;
}

const Vector3F & Triangle::X(int idx) const
{ return P(idx); }

const Vector3F & Triangle::supportPoint(const Vector3F & v, Vector3F * localP) const
{
	float maxdotv = -1e19f;
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

bool Triangle::sampleP(Vector3F & dst, const BoundingBox &  box) const
{
	for(int i=0; i<10; ++i) {
		float a = ((float)(rand() & 1023)) / 1023.f;
		float b = (1.f - a ) * ((float)(rand() & 1023)) / 1023.f;
		float c = 1.f - a - b;
		dst = m_p0 * a + m_p1 * b + m_p2 * c;
		if(box.isPointInside(dst) ) return true;
	}
	return false;
}

Tetrahedron::Tetrahedron()
{}

void Tetrahedron::set(const Vector3F & p0, const Vector3F & p1,
			const Vector3F & p2, const Vector3F & p3)
{ m_p[0] = p0; m_p[1] = p1; m_p[2] = p2; m_p[3] = p3; }

BoundingBox Tetrahedron::calculateBBox() const
{ 
	BoundingBox b;
    b.expandBy(m_p[0]);
    b.expandBy(m_p[1]);
	b.expandBy(m_p[2]);
	b.expandBy(m_p[3]);
    b.expand(b.getLongestDistance() * 1e-4f);
    return b;
}

ShapeType Tetrahedron::ShapeTypeId = TTetrahedron;

std::string Tetrahedron::GetTypeStr()
{ return "tetrahedron"; }

ShapeType Tetrahedron::shapeType() const
{ return TTetrahedron; }

Vector3F Tetrahedron::X(int idx) const
{ return m_p[idx]; }

Vector3F Tetrahedron::supportPoint(const Vector3F & v, Vector3F * localP) const
{
	float maxdotv = -1e19f;
    float dotv;
	int ir = 0;
	
    for(int i=0; i < 4; ++i) {
        dotv = X(i).dot(v);
        if(dotv > maxdotv) {
            maxdotv = dotv;
            ir = i;
        }
    }
    if(localP) *localP = X(ir);
    return X(ir);
}

}

}
//;~