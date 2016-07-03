/*
 *  ClosestSampleTest.cpp
 *  
 *
 *  Created by jian zhang on 7/3/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include "ClosestSampleTest.h"
#include <line_math.h>

using namespace aphid;

namespace ttg {

ClosestSampleTest::ClosestSampleTest(const std::vector<Vector3F> & src)
{
	m_smps = NULL;
	m_N = src.size();
	if(m_N>0) {
		m_smps = new Vector3F[m_N];
		int i=0;
		for(;i<m_N;++i)
			m_smps[i] = src[i];
	}
}

ClosestSampleTest::~ClosestSampleTest()
{
	if(m_smps) delete[] m_smps;
}

int ClosestSampleTest::getClosest(Vector3F & dst, float & d, 
				const Vector3F & toPnt) const
{
	if(m_N < 1) return -1;
	int i=0;
	int r;
	float minD = 1e8f;
	for(int i=0;i<m_N;++i) {
		d = toPnt.distanceTo(m_smps[i]);
		if(d<minD) {
			minD = d;
			dst = m_smps[i];
			r = i;
		}
	}
	d = minD;
	return r;
}

int ClosestSampleTest::getIntersect(aphid::Vector3F & dst, float & d, 
				const aphid::Vector3F & seg1,
				const aphid::Vector3F & seg2) const
{
	if(m_N < 1) return -1;
	const Vector3F ref = firstUp(seg1, seg2);
	Vector3F pos;
	
	int i=0;
	int r = -1;
	float iDot, minD = 1e8f, minDot = 1e8f;
	for(int i=0;i<m_N;++i) {
	
		if(distancePointLineSegment(d, m_smps[i], seg1, seg2) ) {
			projectPointLineSegment(pos, d, m_smps[i], seg1, seg2);
			pos = m_smps[i] - pos;
			pos.normalize();
			
			iDot = pos.dot(ref);
			if(minDot > iDot) {
				minDot = iDot;
			}
			
		if(d<minD) {
			minD = d;
			r = i;
		}
		}
	}
	
	if(minDot > -0.4f)
		return -1;
	
	if(r > -1) {
		d = minD;
		projectPointLineSegment(dst, d, m_smps[r], seg1, seg2);
	}
	
	return r;
}

int ClosestSampleTest::getClosestOnSegment(aphid::Vector3F & dst, float & d, 
				const aphid::Vector3F & seg1,
				const aphid::Vector3F & seg2) const
{
	if(m_N < 1) return -1;
	int i=0;
	int r = -1;
	float iDot, minD = 1e8f;
	for(int i=0;i<m_N;++i) {
	
		if(distancePointLineSegment(d, m_smps[i], seg1, seg2) ) {
		if(d<minD) {
			minD = d;
			r = i;
		}
		}
	}
	
	if(r > -1) {
		d = minD;
		projectPointLineSegment(dst, d, m_smps[r], seg1, seg2);
	}
	
	return r;
}

Vector3F ClosestSampleTest::firstUp(const Vector3F & seg1,
				const Vector3F & seg2) const
{
	Vector3F dv;
	float d;
	int i=0;
	for(;i<m_N;++i) {
		if(distancePointLineSegment(d, m_smps[i], seg1, seg2) ) {
			if(d > 1e-3f) {
				projectPointLineSegment(dv, d, m_smps[i], seg1, seg2);
				dv = m_smps[i] - dv;
				return dv.normal(); 
			}
		}
	}
	return (seg2 - seg1).perpendicular();
}

}