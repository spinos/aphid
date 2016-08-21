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

#define NUM_SAMPLE 257  /// m, the number of evaluation points
#define POLY_MAX_DEG 6  /// n, the highest order polynomial to evaluate
#define INTERVAL_A -1.f
#define INTERVAL_B  1.f
#define DX_SAMPLE 0.0078125f

/// each evaluate has 0 - m th polynomial
	float m_v[NUM_SAMPLE * (POLY_MAX_DEG+1)];
/// integral of pi(x)pi(x)
	float m_pipi[POLY_MAX_DEG+1];
		
	float m_exactEvaluate[NUM_SAMPLE];
	float m_approximateEvaluate[NUM_SAMPLE];

	float m_coeff[POLY_MAX_DEG+1];
	float m_x[NUM_SAMPLE];
	float m_y[NUM_SAMPLE];

/// integral of xpi(x)pi(x)
	float m_xpipi[POLY_MAX_DEG+1];
	
	static const float NormWeightF[POLY_MAX_DEG+1];
	
public:
	LegendreTest();
	virtual ~LegendreTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	void equalSpacingNodes(int n, float * x) const;
	void computeLegendrePolynomials();
/// integral <pi, pi>
	void computePipi();
	
	void evaluateLegendre(int m, int n, float * v) const;
	float legendreValue1(int k, const float & x, const float & pxm1, const float & pxm2) const;
	
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
	
	void computeCoeff(float * coeff, int m, int n, const float * y, const float * v, const float * pipi) const;
	void computeApproximated(float * yhat, int m, const float * coeff) const;
/// http://mathworld.wolfram.com/Gram-SchmidtOrthonormalization.html
/// eq 24 recurrence relation
	void computePix(float * m_v, float * pipi, float * xpipi,
					int m, int n, const float * x) const;
	float computePi(int k, const float * pipi, const float * xpipi, const float & x,
					const float & pm1, const float & pm2) const;
	float integratePipi(const float * v, int k, int m, int n) const;
	float integrateXpipi(const float * v, const float * x, int k, int m, int n) const;
/// http://mathfaculty.fullerton.edu/mathews/n2003/TrapezoidalRuleMod.html
/// finding the area under a curve y = f(x) over [a, b] with m subintervals x0, x1, xm-1
/// x0 = a, xm-1 = b
	float trapezIntegral(const float & a, const float & b, int m, const float * y) const;
	
};

}