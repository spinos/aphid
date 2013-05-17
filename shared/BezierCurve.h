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

class BezierCurve : public BaseCurve {
public:
	BezierCurve();
	virtual ~BezierCurve();
	
	virtual Vector3F interpolate(float param, Vector3F * data) const;
	virtual Vector3F interpolateByKnot(float param, Vector3F * data) const;
private:
	Vector3F calculateBezierPoint(float param, unsigned k0, unsigned k1, Vector3F * data) const;
};
