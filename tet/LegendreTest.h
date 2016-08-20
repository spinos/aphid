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

#define NUM_SAMPLE 65
#define DX_SAMPLE 0.015625f
#define POLY_MAX_DEG 6
#define POLY_MAX_EVA 10

	float m_exactEvaluate[NUM_SAMPLE];
	float m_approximated[NUM_SAMPLE];
/// each evaluate has 0 - m th polynomial
	float m_v[(POLY_MAX_DEG+1) * POLY_MAX_EVA];
	float m_coeff[POLY_MAX_DEG+1];
	float m_x[POLY_MAX_EVA];
	float m_y[POLY_MAX_EVA];
/// integral of pi(x)pi(x)
	float m_pipi[POLY_MAX_DEG+1];
/// integral of xpi(x)pi(x)
	float m_xpipi[POLY_MAX_DEG+1];
/// polynomial samples
	float m_ps[POLY_MAX_DEG+1][NUM_SAMPLE];

public:
	LegendreTest();
	virtual ~LegendreTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	void equalSpacingNodes(int n, float * x) const;
	void computeLegendrePoly();
	float legendreValue1(int m, const float & x, const float & pxm1, const float & pxm2) const;
	
	float evaluateExact(const float & x) const;
/// m max degree of polynomial highest x^m
/// n number of evaluation nodes m+1
/// x evaluate at
/// v output polynomials count m*(n+1)
	void legendreValue(int m, int n, const float * x, float * v) const;
	void findCoefficients(int m, int n, const float * y, const float * v, float * ci) const;
	float discreteIntegral(int n, const float * y, const float * v) const;
	void printLegendreValues(int m, int n, const float * x, float * v) const;
	void printValues(const char * h, int n, const float * y) const;
	void drawEvaluate(const float * y, aphid::GeoDrawer * dr);
	void drawLegendrePoly(int m, aphid::GeoDrawer * dr);
	void computePipi(float * pipi, int m, int n, const float * v) const;
	void computeCoeff(float * coeff, int m, int n, const float * y, const float * v, const float * pipi) const;
	void computeApproximated(float * yhat, int m, const float * coeff) const;
	void computePix(float * m_v, float * pipi, float * xpipi,
					int m, int n, const float * x) const;
	float computePi(int k, const float * pipi, const float * xpipi, const float & x,
					const float & pm1, const float & pm2) const;
	float integratePipi(const float * v, int k, int m, int n) const;
	float integrateXpipi(const float * v, const float * x, int k, int m, int n) const;
	
};

}