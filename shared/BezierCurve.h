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
#include <Primitive.h>
#include <math/bezierSpline.h>

namespace aphid {

class BezierCurve : public BaseCurve {
public:
	BezierCurve();
	virtual ~BezierCurve();

	const Type type() const;
	
	virtual Vector3F interpolate(float param) const;

	void getSegmentSpline(unsigned i, BezierSpline & sp) const;
	float distanceToPoint(const Vector3F & toP, Vector3F & closestP) const;
	
	static void extractSpline(BezierSpline & spline, unsigned i, Vector3F * cvs, unsigned maxInd);
	static bool intersectBox(BezierSpline & spline, const BoundingBox & box);
	static bool intersectTetrahedron(BezierSpline & spline, const Vector3F * tet, const BoundingBox & box);
	static void distanceToPoint(BezierSpline & spline, const Vector3F & pnt, float & minDistance, Vector3F & closestP);
	static float splineLength(BezierSpline & spline);
	
// overrid base curve
	virtual const BoundingBox calculateBBox() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	virtual float length() const;
// overrid geometry
	virtual bool intersectBox(const BoundingBox & box);
	virtual bool intersectTetrahedron(const Vector3F * tet);
	virtual bool intersectBox(unsigned icomponent, const BoundingBox & box);
	virtual bool intersectTetrahedron(unsigned icomponent, const Vector3F * tet);
private:
	void calculateCage(unsigned seg, Vector3F *p) const;
	Vector3F calculateBezierPoint(float t, Vector3F * data) const;
};

}