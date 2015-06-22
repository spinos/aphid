/*
 *  CurveSampler.cpp
 *  testbcc
 *
 *  Created by jian zhang on 6/23/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "CurveSampler.h"
#include <BezierCurve.h>
CurveSampler::CurveSampler() 
{ m_samples = 0; }

CurveSampler::~CurveSampler() 
{ if(m_samples) delete[] m_samples; }

void CurveSampler::begin() 
{
	m_points.clear();
}

void CurveSampler::end() 
{
	m_numSamples = m_points.size();
	if(m_samples) delete[] m_samples;
	m_samples = new Vector3F[m_numSamples];
	std::vector<Vector3F >::const_iterator it = m_points.begin();
	for(int i=0;it!=m_points.end();++it,i++) m_samples[i] = *it;
		
	m_points.clear();
}

void CurveSampler::process(BezierCurve * curve, float groupSize)
{
	const unsigned ns = curve->numSegments();
	const float estimateD = groupSize * .08f;
	
	BezierSpline spl;
	float sl;
	float delta;
	unsigned i=0;
	for(;i<ns;i++) {
		curve->getSegmentSpline(i, spl);
		sl = BezierCurve::splineLength(spl);
		if(sl<estimateD*.02f) {
			std::cout<<" curve sampler warning: segment is too short "<<sl<<"\n";
			continue;
		}
// at least 4 split per segment
		delta = sl * .25f;
		if(delta > estimateD) delta = estimateD;
		sampleSeg(&spl, delta);
	}
	
	m_points.push_back(spl.cv[3]);
	std::cout<<" curve sampler n samples "<<m_points.size();
}

void CurveSampler::sampleSeg(BezierSpline * spl, float delta)
{
	m_points.push_back(spl->cv[0]);
	float curL = delta;
	float param = splineParameterByLength(*spl, curL);
	while(param < .99f) {
		m_points.push_back(spl->calculateBezierPoint(param));
		curL += delta;
		param = splineParameterByLength(*spl, curL);
	}
}

const unsigned CurveSampler::numSamples() const
{ return m_numSamples; }
	
Vector3F * CurveSampler::samples() const
{ return m_samples; }

float CurveSampler::splineParameterByLength(BezierSpline & spline, float expectedLength)
{
	float pmin = 0.f;
	float pmax = 1.f;
	float result = (pmin + pmax) * .5f;
	float lastResult = result;
	BezierSpline a, b, c;
	spline.deCasteljauSplit(a, b);
	
	float l = BezierCurve::splineLength(a);
	while(Absolute(l - expectedLength) > 1e-4) {
		
		if(l > expectedLength) {
			c = a;
			c.deCasteljauSplit(a, b);
			
			pmax = result;
			
			l -= BezierCurve::splineLength(b);
		}
		else {
			c = b;
			c.deCasteljauSplit(a, b);
			
			l += BezierCurve::splineLength(a);
			
			pmin = result;
		}
		
		result = (pmin + pmax) * .5f;
		
		if(Absolute(result - lastResult) < 1e-4) break;
		
		lastResult = result;
	}
	return result;
}
//:~