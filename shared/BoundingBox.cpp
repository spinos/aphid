/*
 *  BoundingBox.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/17/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "Aphid.h"
#include "BoundingBox.h"
#include <Ray.h>
#include <Plane.h>
namespace aphid {

BoundingBox::BoundingBox()
{
	reset();
}

BoundingBox::BoundingBox(const float & x0, const float & y0, const float & z0,
	            const float & x1, const float & y1, const float & z1)
{
    m_data[0] = x0;
    m_data[1] = y0;
    m_data[2] = z0;
    m_data[3] = x1;
    m_data[4] = y1;
    m_data[5] = z1;
}

BoundingBox::BoundingBox(const float * d)
{
    m_data[0] = d[0];
    m_data[1] = d[1];
    m_data[2] = d[2];
    m_data[3] = d[3];
    m_data[4] = d[4];
    m_data[5] = d[5];
}

void BoundingBox::reset()
{
	m_data[0] = m_data[1] = m_data[2] = 10e8;
	m_data[3] = m_data[4] = m_data[5] = -10e8;
}

void BoundingBox::setOne()
{
	setMin(-1.f, -1.f, -1.f);
	setMax(1.f, 1.f, 1.f);
}

void BoundingBox::setMin(float x, int axis)
{
	m_data[axis] = x;
}
	
void BoundingBox::setMax(float x, int axis)
{
	m_data[axis + 3] = x;
}

void BoundingBox::setMin(float x, float y, float z)
{
	m_data[0] = x; m_data[1] = y; m_data[2] = z;
}

void BoundingBox::setMax(float x, float y, float z)
{
	m_data[3] = x; m_data[4] = y; m_data[5] = z;
}

void BoundingBox::update(const Vector3F & p)
{
	updateMin(p);
	updateMax(p);
}

void BoundingBox::updateMin(const Vector3F & p)
{
	if(m_data[0] > p.x) m_data[0] = p.x;
	if(m_data[1] > p.y) m_data[1] = p.y;
	if(m_data[2] > p.z) m_data[2] = p.z;
}

void BoundingBox::updateMax(const Vector3F & p)
{
	if(m_data[3] < p.x) m_data[3] = p.x;
	if(m_data[4] < p.y) m_data[4] = p.y;
	if(m_data[5] < p.z) m_data[5] = p.z;
}

const int BoundingBox::getLongestAxis() const
{
	Vector3F d(m_data[3] - m_data[0], m_data[4] - m_data[1], m_data[5] - m_data[2]);
	if(d.y >= d.x && d.y >= d.z) return 1;
	if(d.z >= d.x && d.z >= d.y) return 2;
	return 0;
}

const float BoundingBox::getLongestDistance() const
{
    Vector3F d(m_data[3] - m_data[0], m_data[4] - m_data[1], m_data[5] - m_data[2]);
	if(d.y >= d.x && d.y >= d.z) return d.y;
	if(d.z >= d.x && d.z >= d.y) return d.z;
	return d.x;
}

const float BoundingBox::getMin(int axis) const
{
	return m_data[axis];
}

const float BoundingBox::getMax(int axis) const
{
	return m_data[axis + 3];
}

const Vector3F BoundingBox::getMin() const { return Vector3F(m_data[0], m_data[1], m_data[2]); }
const Vector3F BoundingBox::getMax() const { return Vector3F(m_data[3], m_data[4], m_data[5]); }

const float BoundingBox::area() const
{
	return ((m_data[3] - m_data[0]) * (m_data[4] - m_data[1]) + (m_data[3] - m_data[0]) * (m_data[5] - m_data[2]) + (m_data[4] - m_data[1]) * (m_data[5] - m_data[2])) * 2.f;
}

const float BoundingBox::crossSectionArea(const int &axis) const
{
	if(axis == 0) {
		return (m_data[4] - m_data[1]) * (m_data[5] - m_data[2]);
	}
	if(axis == 1) {
		return (m_data[3] - m_data[0]) * (m_data[5] - m_data[2]);
	}
	return (m_data[3] - m_data[0]) * (m_data[4] - m_data[1]);
}

const float BoundingBox::distance(const int &axis) const
{
	return m_data[axis + 3] - m_data[axis];
}

void BoundingBox::split(int axis, float pos, BoundingBox & left, BoundingBox & right) const
{
	left = right = *this;
	
	if(axis == 0) {
		left.m_data[3] = pos;
		right.m_data[0] = pos;
	}
	else if(axis == 1) {
		left.m_data[4] = pos;
		right.m_data[1] = pos;
	}
	else {
		left.m_data[5] = pos;
		right.m_data[2] = pos;
	}
}

void BoundingBox::expandBy(const BoundingBox &another)
{
	if(m_data[0] > another.m_data[0]) m_data[0] = another.m_data[0];
	if(m_data[1] > another.m_data[1]) m_data[1] = another.m_data[1];
	if(m_data[2] > another.m_data[2]) m_data[2] = another.m_data[2];
	
	if(m_data[3] < another.m_data[3]) m_data[3] = another.m_data[3];
	if(m_data[4] < another.m_data[4]) m_data[4] = another.m_data[4];
	if(m_data[5] < another.m_data[5]) m_data[5] = another.m_data[5];
}

void BoundingBox::expandBy(const Vector3F & pos)
{
    if(m_data[0] > pos.x) m_data[0] = pos.x;
    if(m_data[1] > pos.y) m_data[1] = pos.y;
    if(m_data[2] > pos.z) m_data[2] = pos.z;
    if(m_data[3] < pos.x) m_data[3] = pos.x;
    if(m_data[4] < pos.y) m_data[4] = pos.y;
    if(m_data[5] < pos.z) m_data[5] = pos.z;
}

void BoundingBox::expandBy(const Vector3F & pos, float r)
{
    if(m_data[0] > pos.x - r) m_data[0] = pos.x - r;
    if(m_data[1] > pos.y - r) m_data[1] = pos.y - r;
    if(m_data[2] > pos.z - r) m_data[2] = pos.z - r;
    if(m_data[3] < pos.x + r) m_data[3] = pos.x + r;
    if(m_data[4] < pos.y + r) m_data[4] = pos.y + r;
    if(m_data[5] < pos.z + r) m_data[5] = pos.z + r;
}

void BoundingBox::shrinkBy(const BoundingBox &another)
{
    if(m_data[0] < another.m_data[0]) m_data[0] = another.m_data[0];
	if(m_data[1] < another.m_data[1]) m_data[1] = another.m_data[1];
	if(m_data[2] < another.m_data[2]) m_data[2] = another.m_data[2];
	
	if(m_data[3] > another.m_data[3]) m_data[3] = another.m_data[3];
	if(m_data[4] > another.m_data[4]) m_data[4] = another.m_data[4];
	if(m_data[5] > another.m_data[5]) m_data[5] = another.m_data[5];
}

void BoundingBox::expand(float val)
{
    m_data[0] -= val;
    m_data[1] -= val;
    m_data[2] -= val;
    m_data[3] += val;
    m_data[4] += val;
    m_data[5] += val;
}

Vector3F BoundingBox::center() const
{
	return Vector3F(m_data[0] * 0.5f + m_data[3] * 0.5f, m_data[1] * 0.5f + m_data[4] * 0.5f, m_data[2] * 0.5f + m_data[5] * 0.5f);
}

char BoundingBox::touch(const BoundingBox & b) const
{
	if(m_data[0] > b.m_data[3] || m_data[3] < b.m_data[0]) return 0;
	if(m_data[1] > b.m_data[4] || m_data[4] < b.m_data[1]) return 0;
	if(m_data[2] > b.m_data[5] || m_data[5] < b.m_data[2]) return 0;
	return 1;
}

float BoundingBox::distanceTo(const Vector3F & pnt) const
{
    if(isPointInside(pnt)) return -1.f;
    
    float dx = getMin(0) - pnt.x;
    dx = dx > 0.f ? dx : 0.f;
    const float dx1 = pnt.x - getMax(0);
    dx = dx > dx1 ? dx : dx1;
    
    float dy = getMin(1) - pnt.y;
    dy = dy > 0.f ? dy : 0.f;
    const float dy1 = pnt.y - getMax(1);
    dy = dy > dy1 ? dy : dy1;
    
    float dz = getMin(2) - pnt.z;
    dz = dz > 0.f ? dz : 0.f;
    const float dz1 = pnt.z - getMax(2);
    dz = dz > dz1 ? dz : dz1;
    
    return sqrt(dx * dx + dy * dy + dz * dz);
}

char BoundingBox::intersect(const BoundingBox & another) const
{
    for(int i=0; i < 3; i++) {
        if(getMin(i) > another.getMax(i)) return 0;
        if(getMax(i) < another.getMin(i)) return 0;    
    }
    return 1;
}

bool BoundingBox::intersect(const BoundingBox & another, BoundingBox * tightBox) const
{
	float a, b, c, d;
	for(int i=0; i < 3; i++) {
		a = getMin(i);
		b = getMax(i);
		c = another.getMin(i);
		d = another.getMax(i);
		if(a > d) return false;
        if(b < c) return false;
/// higher of min
		if(a < c) a = c;
		tightBox->setMin(a, i);
/// lower of max		
		if(b > d) b = d;
		tightBox->setMax(b, i);    
	}
	return true;
}

char BoundingBox::intersect(const Ray &ray, float *hitt0, float *hitt1) const 
{
    float t0 = ray.m_tmin, t1 = ray.m_tmax;
    for (int i = 0; i < 3; ++i) {
		const float diri = ray.m_dir.comp(i);
		const Vector3F o = ray.m_origin;
		if(IsValueNearZero(diri)) {
			if(i == 0) {
				if(o.x < m_data[0] || o.x > m_data[3]) return 0;
			}
			else if(i == 1) {
				if(o.y < m_data[1] || o.y > m_data[4]) return 0;
			}
			else {
				if(o.z < m_data[2] || o.z > m_data[5]) return 0;
			}
			continue;
		}
        // Update interval for _i_th bounding box slab
        float invRayDir = 1.f / ray.m_dir.comp(i);
        float tNear = (getMin(i) - ray.m_origin.comp(i)) * invRayDir;
        float tFar  = (getMax(i) - ray.m_origin.comp(i)) * invRayDir;

        // Update parametric interval from slab intersection $t$s
        if (tNear > tFar) SwapValues(tNear, tFar);
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar  < t1 ? tFar  : t1;
        if (t0 > t1) return 0;
    }
    if (hitt0) *hitt0 = t0;
    if (hitt1) *hitt1 = t1;
    return 1;
}

char BoundingBox::isPointInside(const Vector3F & p) const
{
	if(p.x < getMin(0) || p.x > getMax(0)) return 0;
	if(p.y < getMin(1) || p.y > getMax(1)) return 0;
	if(p.z < getMin(2) || p.z > getMax(2)) return 0;
	return 1;
}

char BoundingBox::isPointAround(const Vector3F & p, float threshold) const
{
	if(p.x + threshold < getMin(0) || p.x - threshold > getMax(0)) return 0;
	if(p.y + threshold < getMin(1) || p.y - threshold > getMax(1)) return 0;
	if(p.z + threshold < getMin(2) || p.z - threshold > getMax(2)) return 0;
	return 1;
}

char BoundingBox::isBoxAround(const BoundingBox & b, float threshold) const
{
    if(b.getMin(0) > getMax(0) + threshold) return 0;
    if(b.getMax(0) < getMin(0) - threshold) return 0;
    if(b.getMin(1) > getMax(1) + threshold) return 0;
    if(b.getMax(1) < getMin(1) - threshold) return 0;
    if(b.getMin(2) > getMax(2) + threshold) return 0;
    if(b.getMax(2) < getMin(2) - threshold) return 0;
    return 1;
}

char BoundingBox::inside(const BoundingBox & b) const
{
	return (getMin(0) >= b.getMin(0) &&
			getMin(1) >= b.getMin(1) &&
			getMin(2) >= b.getMin(2) &&
			getMax(0) <= b.getMax(0) &&
			getMax(1) <= b.getMax(1) &&
			getMax(2) <= b.getMax(2));
}

char BoundingBox::isValid() const
{
	return (getMin(0) < getMax(0) && getMin(1) < getMax(1) && getMin(2) < getMax(2));
}

/*
// z0
// 2 - 3
// |   |
// 0 - 1
// 
// z1
// 6 - 7
// |   |
// 4 - 5
*/

const Vector3F BoundingBox::corner(const int & i) const
{
    const int iz = i / 4;
    const int iy = (i - iz * 4) / 2;
    const int ix = i - iy * 2 - iz * 4;
    return Vector3F(m_data[3 * ix], m_data[1 + 3 * iy], m_data[2 + 3 * iz]);
}

bool BoundingBox::intersect(const Plane & p, float & tmin, float & tmax) const
{
    float t;
    tmin = 10e8;
    tmax = -10e8;
    for(int i = 0; i < 8; i++) {
        t = p.pointTo(corner(i));
        if(t < tmin) tmin = t;
        if(t > tmax) tmax = t;
    }
    
    if(tmin > 0.f || tmax < 0.f) return false;
    return true;
}

const Vector3F BoundingBox::normal(const int & i) const
{
    if(i<1) return Vector3F::XAxis;
    if(i<2) return Vector3F::YAxis;
    return Vector3F::ZAxis;
}

float BoundingBox::radius() const
{
    float dx = distance(0) * .5f;
    float dy = distance(1) * .5f;
    float dz = distance(2) * .5f;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

float BoundingBox::radiusXZ() const
{
	float dx = distance(0) * .5f;
    float dz = distance(2) * .5f;
    return sqrt(dx*dx + dz*dz);
}

int BoundingBox::numPoints() const
{ return 8; }

Vector3F BoundingBox::X(int idx) const
{
    Vector3F r(m_data[0], m_data[1], m_data[2]);
    if(idx > 3) r.z = m_data[5];
    if((idx & 3) > 1) r.y = m_data[4];
    if(idx & 1) r.x = m_data[3];
    return r;
}

Vector3F BoundingBox::supportPoint(const Vector3F & v, Vector3F * localP) const
{
    float maxdotv = -1e8f;
    float dotv;
	
    Vector3F res, q;
    for(int i=0; i < numPoints(); i++) {
        q = X(i);
        dotv = q.dot(v);
        if(dotv > maxdotv) {
            maxdotv = dotv;
            res = q;
            if(localP) *localP = q;
        }
    }
    
    return res;
}

void BoundingBox::verbose() const
{
	std::cout<<str()<<"\n";
}

void BoundingBox::verbose(const char * pref) const
{
	std::cout<<pref<<str()<<"\n";
}

const std::string BoundingBox::str() const
{
	std::stringstream sst;
	sst.str("");
    sst<<"(("<<m_data[0]<<","<<m_data[1]<<","<<m_data[2]<<"),("<<m_data[3]<<","<<m_data[4]<<","<<m_data[5]<<"))";
	return sst.str();
}

const float * BoundingBox::data() const
{ return m_data; }

}
//:~