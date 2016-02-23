/*
 *  LineDrawer.h
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include "BaseDrawer.h"
#include "ConvexShape.h"

namespace aphid {

class LineBuffer;
class AdaptableStripeBuffer;
class BaseCurve;
struct BezierSpline;
class BezierCurve;
class LineDrawer : public BaseDrawer {
public:
	LineDrawer();
	virtual ~LineDrawer();
	void drawLineBuffer(LineBuffer * line) const;
	void lines(const std::vector<Vector3F> & vs);
	void lineStripes(const unsigned & num, unsigned * nv, Vector3F * vs) const;
	void lineStripes(const unsigned & num, unsigned * nv, Vector3F * vs, Vector3F * cs) const;
	void stripes(AdaptableStripeBuffer * data, const Vector3F & eyeDirection) const;
	void linearCurve(const BaseCurve & curve) const;
	void smoothCurve(const BezierCurve & curve, short deg) const;
	void smoothCurve(const BezierSpline & sp, short deg) const;
	void frustum(const Frustum * f);
};

}
