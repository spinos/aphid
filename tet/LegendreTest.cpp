/*
 *  LegendreTest.cpp
 *  foo
 *  approximation of f(x) within interval [-1, 1]
 *  F(x) = c0*P(0,x) + c1*P(1,x) + c2*P(2,x) + c3*P(3,x) ... + cn*P(n,x), where
 *  ci = integral(f(x)*P(i,x))
 *  P(i,x) is i-th normalized Legendre polynomal
 *
 *  http://mathfaculty.fullerton.edu/mathews/n2003/LegendrePolyMod.html
 *  http://faculty.washington.edu/finlayso/ebook/bvp/OC/Legendre.htm
 *  http://www.mhtlab.uwaterloo.ca/courses/me755/web_chap5.pdf
 *  http://archive.lib.msu.edu/crcmath/math/math/l/l175.htm
 *  
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "LegendreTest.h"
#include <GridTables.h>
#include <Calculus.h>
#include <iostream>
#include <iomanip>

using namespace aphid;
namespace ttg {

LegendreTest::LegendreTest() 
{}

LegendreTest::~LegendreTest() 
{}
	
const char * LegendreTest::titleStr() const
{ return "Legendre Polynomial Approximation Test"; }

float LegendreTest::evaluateExact(const float & x) const
{ 
	//return 10.f - x* 2.f + x * x / 10.f;
	//return x*x*x*x - x*x * 2 / 3 + 1.0 / 9.0;
	//return x*x*x - x * 3.0 / 5.0;
	//return x*x*x*x*x*x - x*x*x*x * 6.0 / 5.0 + x*x * 9.0 / 25.0;
	//return (x*x - 1.0/3.0);
	//return exp(x) * (x*x - 1.0 / 3.0);
	//return exp(x) * (x*x*x - 3.0 *x / 5.0);
	return 1.31f + .26 * exp(x*.785) * sin(x*2.34 - 2.f)+ .03125*sin(x*6); 
	return exp(x); 
}

bool LegendreTest::init()
{
	int i;
	for(i=0;i<M_NUM_EVAL;++i) {
		m_exactEvaluate[i] = evaluateExact(INTERVAL_A + DX_SAMPLE * i);
	}
	
	int m = 5, n = 4;
	std::cout<<"\n deg "<<n<<" measurement "<<m;
	
	computeCoeff(m_coeff, n);
	computeApproximated(m_approximateEvaluate, n, m_coeff);
	
	std::cout<<"\n done!";
	std::cout.flush();
	return true;
}

void LegendreTest::computeCoeff(float * coeff, int n) const
{
#define N_ORD 5
	float Xi[N_ORD];
	float Wi[N_ORD];
	calc::gaussQuadratureRule(N_ORD, Wi, Xi);
	
	float Pi[N_ORD * (n+1)];
	calc::legendreRule(N_ORD, n, Pi, Xi);
	
	float Yi[N_ORD];
	
	int i, j;
	for(i=0; i<=n; ++i) {
		for(j=0;j<N_ORD;++j) {
/// f(x)P(i,x)
			Yi[j] = evaluateExact(Xi[j]) * calc::LegendrePolynomial::P(i, Xi[j]);
		}
			
		coeff[i] = calc::gaussQuadratureRuleIntegrate(1, N_ORD, Xi, Wi, Yi);
		
		std::cout<<"\n  c"<<i<<" "<<coeff[i];
	}
}

void LegendreTest::computeApproximated(float * yhat, int n, const float * coeff) const
{
	int i, j;
	for(i=0;i<M_NUM_EVAL;++i) {
		yhat[i] = 0.f;
		for(j=0; j<=n; ++j) {
			yhat[i] += coeff[j] * calc::LegendrePolynomial::P(j, INTERVAL_A + DX_SAMPLE * i);
		}
	}
}

void LegendreTest::draw(GeoDrawer * dr)
{
	glColor3f(0.f, 0.f, 0.f);
	drawEvaluate(m_exactEvaluate, dr);
	
	glColor3f(0.33f, 1.f, 0.33f);
	drawEvaluate(m_approximateEvaluate, dr);

#if 1
	Vector3F mcol;
	int i=0;
	for(;i<=POLY_MAX_DEG;++i) {
		sdb::gdt::GetCellColor(mcol, i);
		glColor3fv((const float *)&mcol);
		drawLegendrePoly(i, dr);
	}
#endif
}

void LegendreTest::drawEvaluate(const float * y, GeoDrawer * dr)
{
	glPushMatrix();
	glScaled(15.0, 10.0, 10.0);
	glBegin(GL_LINES);
	int i;
	for(i=0;i<M_NUM_EVAL-1;++i) {
		glVertex3f(INTERVAL_A + DX_SAMPLE * i, y[i], 0.f);
		glVertex3f(INTERVAL_A + DX_SAMPLE * (i+1), y[i+1], 0.f);
	}
	glEnd();
	glPopMatrix();
}

void LegendreTest::drawLegendrePoly(int m, GeoDrawer * dr)
{
	glPushMatrix();
	glScaled(15.0, 10.0, 10.0);
	glBegin(GL_LINES);
	int i;
	float norm, x;
	for(i=0;i<M_NUM_EVAL-1;++i) {
		norm = calc::LegendrePolynomial::norm(m);
		x = INTERVAL_A + DX_SAMPLE * i;
		glVertex3f(x, calc::LegendrePolynomial::P(m, x) * norm, 0.f);
		x += DX_SAMPLE;
		glVertex3f(x, calc::LegendrePolynomial::P(m, x) * norm, 0.f);
	}
	glEnd();
	glPopMatrix();
}

}