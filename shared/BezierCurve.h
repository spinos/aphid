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
	
	virtual Vector3F interpolate(float param, Vector3F * data);
	virtual Vector3F interpolateByKnot(float param, Vector3F * data);
	virtual Vector3F interpolate(Vector3F * data) const;
	
	virtual void calculateT(float param);
	unsigned m_k00, m_k11;
private:
	void fourControlKnots();
	Vector3F calculateBezierPoint(float t, unsigned k00, unsigned k0, unsigned k1, unsigned k11, Vector3F * data) const;
};
