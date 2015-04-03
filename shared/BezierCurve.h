/*
 *  BezierCurve.h
 *  knitfabric
 *
 *  Created by jian zhang on 5/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <BaseCurve.h>

struct SimpleBezierSpline {
    void deCasteljauSplit(SimpleBezierSpline & a, SimpleBezierSpline & b)
    {
        Vector3F d = cv[1] * .5f + cv[2] * .5f;
        a.cv[0] = cv[0];
        a.cv[1] = cv[0] * .67f + cv[1] * .33f;
        a.cv[2] = a.cv[1] * .33f + d * .67f; 
        
        b.cv[3] = cv[3];
        b.cv[2] = cv[3] * .67f + cv[2] * .33f;
        b.cv[1] = b.cv[2] * .33f + d * .67f;
        
        a.cv[3] = b.cv[0] = a.cv[2] * .5f + b.cv[1] * .5f;
    }
    
    unsigned segmentByParameter(float t) const
    {
        if(t <= 0.f) return 0;
        if(t >= 1.f) return 2;
        return t * 3;
    }
    
    Vector3F calculateBezierPoint(float param) const
    {
        unsigned seg = segmentByParameter(param);
        
        float t = param * 3 - seg;
        
        float u = 1.f - t;
        float tt = t * t;
        float uu = u*u;
        float uuu = uu * u;
        float ttt = tt * t;
        
        Vector3F p0, p3;
        
        if(seg == 0) p0 = cv[0];
        else p0 = (cv[seg] * 2.f + cv[seg - 1] + cv[seg + 1]) * .25f;

        if(seg >= 2) p3 = cv[3];
        else p3 = (cv[seg + 1] * 2.f + cv[seg] + cv[seg + 2]) * .25f;

        Vector3F p1 = (cv[seg + 1] * 0.5f + cv[seg] * 0.5f) * .5f + p0 * .5f;
        Vector3F p2 = (cv[seg] * 0.5f + cv[seg + 1] * 0.5f) * .5f + p3 * .5f;
    
        Vector3F p = p0 * uuu; //first term
        p += p1 * 3.f * uu * t; //second term
        p += p2 * 3.f * u * tt; //third term
        p += p3 * ttt; //fourth term
        return p;
    }
    
    Vector3F cv[4];
};

class BezierCurve : public BaseCurve {
public:
	BezierCurve();
	virtual ~BezierCurve();
	
	virtual Vector3F interpolate(float param) const;

	void getAccSegmentCurves(BezierCurve * dst) const;
	void getAccSegmentSpline(unsigned i, SimpleBezierSpline & sp) const;
	float distanceToPoint(const Vector3F & toP, Vector3F & closestP) const;
private:
	void calculateCage(unsigned seg, Vector3F *p) const;
	Vector3F calculateBezierPoint(float t, Vector3F * data) const;
	void distanceToPoint(SimpleBezierSpline & spline, const Vector3F & pnt, float & minDistance, Vector3F & closestP) const;
	
};
