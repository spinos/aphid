/*
 *  Fiber.cpp
 *  
 *
 *  Created by jian zhang on 9/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "Fiber.h"
#include "SegmentNormals.h"
#include <math/Ray.h>
#include <math/Matrix44F.h>

namespace aphid {

FiberBulder::FiberBulder()
{}

FiberBulder::~FiberBulder()
{ m_pnts.clear(); }

void FiberBulder::begin()
{ m_pnts.clear(); }

void FiberBulder::addPoint(const Vector3F& pv)
{ m_pnts.push_back(pv); }

void FiberBulder::setPoint(int i, const Vector3F& pv)
{ m_pnts[i] = pv; }

void FiberBulder::insertPoint(int i, const Vector3F& pv)
{ m_pnts.insert(m_pnts.begin() + i, pv); }

void FiberBulder::end()
{ 
	const int ns = numPoints() - 1;
	m_normals.reset(new SegmentNormals(ns) );
	
	Vector3F p0p1 = m_pnts[1] - m_pnts[0];
	Vector3F ref(-1.f, 0.f, 0.f);
	float vy = p0p1.normal().y;
	if(vy < 0.f)
		vy = -vy;
	if(vy < .1f)
		ref.set(0.f, 1.f, 0.f);
	
	m_normals->calculateFirstNormal(p0p1, ref);
	
	for(int i=1;i<ns;++i) {
		Vector3F p0 = m_pnts[i - 1];
		Vector3F p1 = m_pnts[i];
		Vector3F p2 = m_pnts[i+1];
		p0p1 = p1 - p0;
		Vector3F p1p2 = p2 - p1;
		Vector3F p1pm02 = (p0 + p2) * 0.5f - p1;
		m_normals->calculateNormal(i, p0p1, p1p2, p1pm02 );
	}
	
}

int FiberBulder::numPoints() const
{ return m_pnts.size(); }

int FiberBulder::numSegments() const
{ return numPoints() - 1; }

const Vector3F& FiberBulder::getPoint(int i) const
{ return m_pnts[i]; }

const Vector3F& FiberBulder::getNormal(int i) const
{ return m_normals->getNormal(i); }

void FiberBulder::copyStrand(OrientedPoint* pnts,
				FiberUnit* segs) const
{
	const int np = numPoints();
	
	for(int i=0;i<np-1;++i) {
		segs[i]._v0 = &pnts[i];
		segs[i]._v1 = &pnts[i + 1];
	}
	
	for(int i=0;i<np;++i) {
		pnts[i]._x = m_pnts[i];
		pnts[i]._x0 = pnts[i]._x;
	}
	
	for(int i=0;i<np;++i) {		
		if(i==0) {
			setRotation(&pnts[i],
					pnts[1]._x - pnts[0]._x, getNormal(0), 0);
		} else if(i==np-1) {
			setRotation(&pnts[i],
					pnts[i]._x - pnts[i-1]._x, getNormal(i-1), i);
		} else {
			setRotation(&pnts[i],
					pnts[i+1]._x - pnts[i-1]._x, 
						getNormal(i) + getNormal(i-1), 
						i);
		}
	}
}

void FiberBulder::setRotation(OrientedPoint* dst,
				const Vector3F& vx, 
				const Vector3F& vy, const int& i) const
{
	Vector3F r0 = vx.normal();
	Vector3F r2 = r0.cross(vy);
	r2.normalize();
	Vector3F r1 = r2.cross(r0);
	r1.normalize();
	Matrix33F rotm(r0, r1, r2);
	rotm.getQuaternion(dst->_q);
}


Fiber::Fiber() :
m_numPnts(0)
{}

Fiber::~Fiber()
{}

void Fiber::create(const FiberBulder* bld,
					OrientedPoint* pnt,
					FiberUnit* seg)
{
	m_pnts = pnt;
	m_segs = seg;
	m_numPnts = bld->numPoints();
	
	update();	
}

void Fiber::dump(FiberBulder* bld) const
{
	for(int i=0;i<numPoints();++i) {
		if(bld->numPoints() <= i)
			bld->addPoint(m_pnts[i]._x);
		else
			bld->setPoint(i, m_pnts[i]._x);
	}
}

void Fiber::initialize()
{
	const int& np = numPoints();
	for(int i=0;i<np;++i) {
		m_pnts[i]._x0 = m_pnts[i]._x;
		if(i==0) {
			setRotation(m_pnts[1]._x - m_pnts[0]._x, getPointUp(0), 0);
		} else if(i==np-1) {
			setRotation(m_pnts[i]._x - m_pnts[i-1]._x, getPointUp(i-1), i);
		} else {
			setRotation(m_pnts[i+1]._x - m_pnts[i-1]._x, 
						getPointUp(i) + getPointUp(i-1), 
						i);
		}
	}
	update();
}

const int& Fiber::numPoints() const
{ return m_numPnts; }

int Fiber::numSegments() const
{ return m_numPnts - 1; }

OrientedPoint* Fiber::points()
{ return m_pnts; }

FiberUnit* Fiber::segments()
{ return m_segs; }

const OrientedPoint* Fiber::points() const
{ return m_pnts; }

const FiberUnit* Fiber::segments() const
{ return m_segs; }

void Fiber::setPoint(const Vector3F& pv, const int& i)
{ 
	m_pnts[i]._x = pv; 
	m_pnts[i]._x0 = pv;
}

void Fiber::setRotation(const Vector3F& vx, 
				const Vector3F& vy, const int& i)
{
	Vector3F r0 = vx.normal();
	Vector3F r2 = r0.cross(vy);
	r2.normalize();
	Vector3F r1 = r2.cross(r0);
	r1.normalize();
	Matrix33F rotm(r0, r1, r2);
	rotm.getQuaternion(m_pnts[i]._q);
}

void Fiber::update()
{
	const int n = numSegments();
	for(int i=0;i<n;++i) {
	
		FiberUnit& si = m_segs[i];
		
		float ul = .789f * si._v0->_x.distanceTo(si._v1->_x);
		
		Matrix33F r0(si._v0->_q);
		r0.getSide(si._tng0);
		si._tng0 *= ul;
		
		Matrix33F r1(si._v1->_q);
		r1.getSide(si._tng1);
		si._tng1 *= ul;
	}
}

void Fiber::InterpolateSpace(Matrix44F& mat, 
					float& segl,
					const FiberUnit& fu,
					const float& t)
{
	Vector3F p0;
	InterpolatePoint(p0, fu, t);
	
	Vector3F p1;
	InterpolatePoint(p1, fu, t + .1249f);
	
	Vector3F dp = p1 - p0;
	segl = dp.length();
	dp /= segl;
	
	Quaternion q;
	Quaternion::Slerp(q, fu._v0->_q, fu._v1->_q, t);
	Matrix33F rotm(q);
	rotm.rotateSideTo(dp);
	mat.setRotation(rotm);
	mat.setTranslation(p0);
}

void Fiber::InterpolatePoint(Vector3F& pv,
					const FiberUnit& fu,
					const float& t)
{
	if(t<1e-2f) {
		pv = fu._v0->_x;
		return;
	} else if(t>.99f) {
		pv = fu._v1->_x;
		return;
	}
	
	float s2 = t * t;
	float s3 = s2 * t;
	float h1 =  2.f * s3 - 3.f * s2 + 1.f;          // calculate basis function 1
	float h2 = -2.f * s3 + 3.f * s2;              // calculate basis function 2
	float h3 =   s3 - 2.f * s2 + t;         // calculate basis function 3
	float h4 =   s3 -  s2;              // calculate basis function 4
	
	pv = (fu._v0->_x * h1 +                    // multiply and sum all funtions
             fu._v1->_x * h2 +                    // together to build the interpolated
             fu._tng0 * h3 +                    // point along the curve.
             fu._tng1 * h4);
}

Vector3F Fiber::getPointUp(const int& i) const
{
	Matrix33F rotm(m_pnts[i]._q);
	Vector3F r;
	rotm.getUp(r);
	return r;
}

}
