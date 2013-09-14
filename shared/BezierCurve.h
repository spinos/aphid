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
	
	virtual Vector3F interpolate(float param) const;

private:
	void calculateCage(unsigned seg, Vector3F *p) const;
	Vector3F calculateBezierPoint(float t, Vector3F * data) const;
};
