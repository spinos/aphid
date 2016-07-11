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
	
	float la, lb;
	Vector3F pa, pb, va, vb, mid;
	
	getClosest(pa, d, seg1);
	va = pa - seg1;
	la = d;
	va.normalize();
	
	getClosest(pb, d, seg2);
	vb = pb - seg2;
	lb = d;
	vb.normalize();
	
/// both end on the same side
	if(va.dot(vb) > 0.f)
		return -1;
	
/// close to weighted average		
	getClosest(mid, d, pa * lb / (la + lb)  + pb * la / (la + lb) );
	
	if(!distancePointLineSegment(d, mid, seg1, seg2) )
		return -1;
	
/// p on seg
	projectPointLineSegment(dst, d, mid, seg1, seg2);
	
	return 1;
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

}