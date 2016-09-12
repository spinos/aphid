/*
 *  LegendreTest.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Scene.h"

namespace ttg {

class LegendreTest : public Scene {

#define M_NUM_EVAL 257  /// m, the number of evaluation points
#define POLY_MAX_DEG 6  /// n, the highest order polynomial to evaluate
#define INTERVAL_A -1.f
#define INTERVAL_B  1.f
#define DX_SAMPLE 0.0078125f
		
	float m_exactEvaluate[M_NUM_EVAL];
	float m_approximateEvaluate[M_NUM_EVAL];
	float m_coeff[POLY_MAX_DEG+1];
		
public:
	LegendreTest();
	virtual ~LegendreTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	float evaluateExact(const float & x) const;
	void drawEvaluate(const float * y, aphid::GeoDrawer * dr);
	void drawLegendrePoly(int m, aphid::GeoDrawer * dr);
/// n max degree of evaluate
	void computeCoeff(float * coeff, int n) const;
	void computeApproximated(float * yhat, int n, const float * coeff) const;
	
};

}