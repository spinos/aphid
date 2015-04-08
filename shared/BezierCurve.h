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

struct BezierSpline {
    void deCasteljauSplit(BezierSpline & a, BezierSpline & b)
    {
        Vector3F d = cv[1] * .5f + cv[2] * .5f;
        a.cv[0] = cv[0];
        a.cv[1] = cv[0] * .5f + cv[1] * .5f;
        a.cv[2] = a.cv[1] * .5f + d * .5f; 
        
        b.cv[3] = cv[3];
        b.cv[2] = cv[3] * .5f + cv[2] * .5f;
        b.cv[1] = b.cv[2] * .5f + d * .5f;
        
        a.cv[3] = b.cv[0] = a.cv[2] * .5f + b.cv[1] * .5f;
    }
    
    Vector3F calculateBezierPoint(float t) const
    {
        float u = 1.f - t;
        float tt = t * t;
        float uu = u*u;
        float uuu = uu * u;
        float ttt = tt * t;
        
        Vector3F p = cv[0] * uuu; //first term
        p += cv[1] * 3.f * uu * t; //second term
        p += cv[2] * 3.f * u * tt; //third term
        p += cv[3] * ttt; //fourth term
        return p;
    }
    
    bool straightEnough() const
    {
        const float d = cv[0].distanceTo(cv[3]);
        return ( ( cv[0].distanceTo(cv[1]) + cv[1].distanceTo(cv[2]) + cv[2].distanceTo(cv[3]) - d ) / d ) < .005f;
    }
    
    Vector3F cv[4];
};

class BezierCurve : public BaseCurve {
public:
	BezierCurve();
	virtual ~BezierCurve();
	
	virtual Vector3F interpolate(float param) const;

	void getSegmentSpline(unsigned i, BezierSpline & sp) const;
	float distanceToPoint(const Vector3F & toP, Vector3F & closestP) const;
	bool intersectBox(const BoundingBox & box) const;
	bool intersectTetrahedron(const Vector3F * tet) const;
private:
	void calculateCage(unsigned seg, Vector3F *p) const;
	Vector3F calculateBezierPoint(float t, Vector3F * data) const;
	void distanceToPoint(BezierSpline & spline, const Vector3F & pnt, float & minDistance, Vector3F & closestP) const;
	bool intersectBox(BezierSpline & spline, const BoundingBox & box) const;
	bool intersectTetrahedron(BezierSpline & spline, const Vector3F * tet, const BoundingBox & box) const;
};
