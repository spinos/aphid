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
class LineBuffer;
class AdaptableStripeBuffer;
class BaseCurve;
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
	void smoothCurve(BaseCurve & curve, short deg) const;
};
