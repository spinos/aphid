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

namespace aphid {

namespace cvx {
class Frustum;

}

class LineBuffer;
class AdaptableStripeBuffer;
class BaseCurve;
struct BezierSpline;
class BezierCurve;
class LineDrawer : public BaseDrawer {

	Vector3F m_alignDir;
	
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
	void drawNumber(int x, const Vector3F & p, float scale = 1.f) const;
	void drawDigit(int d) const;
	void setAlignDir(const Vector3F & v);
	void frustum(const cvx::Frustum * f);
	
protected:
	const Vector3F & alignDir() const;
	
private:
	static float DigitLineP[10][8][2];
	static int DigitM[9];
};

}
