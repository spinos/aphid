/*
 *  CurveSampler.h
 *  testbcc
 *
 *  Created by jian zhang on 6/23/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <AllMath.h>
#include <vector>
class BezierCurve;
struct BezierSpline;
class CurveSampler {
public:
	CurveSampler();
	virtual ~CurveSampler();
	
	void begin();
	void end();
	void process(BezierCurve * curve, float groupSize);
	
	static float CurveSampler::splineParameterByLength(BezierSpline & spline, float expectedLength);
	
	const unsigned numSamples() const;
	Vector3F * samples() const;
private:
	void sampleSeg(BezierSpline * spl, float delta);
private:
	std::vector<Vector3F > m_points;
	unsigned m_numSamples;
	Vector3F * m_samples;
};