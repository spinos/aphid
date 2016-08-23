/*
 *  Legendre2DTest.cpp
 *  foo
 *  
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Legendre2DTest.h"
#include <GridTables.h>
#include <Calculus.h>
#include <iostream>
#include <iomanip>

using namespace aphid;
namespace ttg {

Legendre2DTest::Legendre2DTest() 
{}

Legendre2DTest::~Legendre2DTest() 
{}
	
const char * Legendre2DTest::titleStr() const
{ return "2D Legendre Polynomial Approximation"; }

float Legendre2DTest::evaluateExact(const float & x) const
{ 
	//return 10.f - x* 2.f + x * x / 10.f;
	//return x*x*x*x - x*x * 2 / 3 + 1.0 / 9.0;
	//return x*x*x - x * 3.0 / 5.0;
	//return x*x*x*x*x*x - x*x*x*x * 6.0 / 5.0 + x*x * 9.0 / 25.0;
	//return (x*x - 1.0/3.0);
	//return exp(x) * (x*x - 1.0 / 3.0);
	//return exp(x) * (x*x*x - 3.0 *x / 5.0);
	return 1.1f + .4 * exp(x*.5) * cos(x*3); 
	return exp(x); 
}

bool Legendre2DTest::init()
{
	int i;
	for(i=0;i<M_NUM_EVAL;++i) {
		m_exactEvaluate[i] = evaluateExact(INTERVAL_A + DX_SAMPLE * i);
	}
	
	//std::cout<<"\n test integral "<<calc::trapezIntegral(INTERVAL_A, INTERVAL_B, M_NUM_EVAL, m_exactEvaluate);
	
	calc::legendreRules(M_NUM_EVAL, POLY_MAX_DEG, m_v, INTERVAL_A, INTERVAL_B);

	computePipi();
	
	int m = 7, n = 5;
	std::cout<<"\n deg "<<n<<" sample "<<m;
	
	const float dx = (INTERVAL_B - INTERVAL_A) / (float)(m-1);
	
	float * X = new float[m];
	float * Y = new float[m];
	float * Px = new float[m];
	
	for(i=0;i<m;++i) {
		X[i] = INTERVAL_A + dx * i;
		Y[i] = evaluateExact(X[i]);
	}
	
	printValues("X", m, X);
	printValues("Y", m, Y);
#if 0
	for(i=0;i<m;++i) {
		Px[i] = samplePolyValue(0, X[i]);
	}
	printValues("P0(x)", m, Px);
	
	for(i=0;i<m;++i) {
		Px[i] = samplePolyValue(1, X[i]);
	}
	printValues("P1(x)", m, Px);
	
	for(i=0;i<m;++i) {
		Px[i] = samplePolyValue(2, X[i]);
	}
	printValues("P2(x)", m, Px);
	
	for(i=0;i<m;++i) {
		Px[i] = samplePolyValue(3, X[i]);
	}
	printValues("P3(x)", m, Px);
#endif
/// interpolate y through out samples
	for(i=0;i<M_NUM_EVAL;++i) {
		m_approximateEvaluate[i] = calc::interpolate(m, Y, M_NUM_EVAL, INTERVAL_A + DX_SAMPLE * i,
										INTERVAL_A, INTERVAL_B, DX_SAMPLE);
	}

	computeCoeff(m_coeff, n, m_approximateEvaluate, m_pipi);

	computeApproximated(m_approximateEvaluate, n, m_coeff);
	
	delete[] X;
	delete[] Y;
	delete[] Px;
	std::cout<<"\n done!";
	std::cout.flush();
	return true;
}

float Legendre2DTest::samplePolyValue(int i, float x) const
{
	int j = (x - INTERVAL_A) / DX_SAMPLE;
	return m_v[M_NUM_EVAL * i + j];
}

void Legendre2DTest::computePipi()
{
	float px2[M_NUM_EVAL];
	int i, j;
	for(i=0; i<=POLY_MAX_DEG; ++i) {
		for(j=0;j<M_NUM_EVAL; ++j) {
			px2[j] = m_v[M_NUM_EVAL * i + j];
			px2[j] *= px2[j]; /// <pi, pi>
		}
		
		m_pipi[i] = calc::trapezIntegral(INTERVAL_A, INTERVAL_B, M_NUM_EVAL, px2);
			
		std::cout<<"\n <p"<<i<<",p"<<i<<"> "<<m_pipi[i];
	}
	
#if 0
	std::cout<<"\n test orthogonality";
	
	for(j=0;j<M_NUM_EVAL; ++j) {
			px2[j] = m_v[M_NUM_EVAL * 1 + j];
			px2[j] *= m_v[M_NUM_EVAL * 2 + j]; /// <pi, pi>
	}
	
	std::cout<<"\n <p2,p1> "<<calc::trapezIntegral(INTERVAL_A, INTERVAL_B, M_NUM_EVAL, px2);
#endif

}

void Legendre2DTest::computeCoeff(float * coeff, int n, const float * y,
						const float * pipi) const
{
	float Px[M_NUM_EVAL];
	
	int i, j;
	for(i=0; i<=n; ++i) {
		for(j=0;j<M_NUM_EVAL;++j) {
			Px[j] = m_v[M_NUM_EVAL * i + j];
			Px[j] *= y[j];
		}
			
		coeff[i] = calc::trapezIntegral(INTERVAL_A, INTERVAL_B, M_NUM_EVAL, Px);
		
		//std::cout<<"\n c"<<i<<" "<<coeff[i];
		coeff[i] /= pipi[i];
		
		std::cout<<"\n  c"<<i<<" "<<coeff[i];
	}
}

void Legendre2DTest::computeApproximated(float * yhat, int n, const float * coeff) const
{
	int i, j;
	for(i=0;i<M_NUM_EVAL;++i) {
		yhat[i] = 0.f;
		for(j=0; j<=n; ++j) {
			yhat[i] += coeff[j] * m_v[j*M_NUM_EVAL + i];// * calc::LegendreNormWeightF[j];
		}
	}
}

void Legendre2DTest::draw(GeoDrawer * dr)
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

void Legendre2DTest::printValues(const char * h, int n, const float * y) const
{
	std::cout<<"\n "<<std::setw(8)<<h;
	int i=0;
	for(;i<n;++i)
		std::cout<<" "<<std::setw(12) <<std::setprecision(8)<<y[i];
}

float Legendre2DTest::discreteIntegral(int n, const float * y, const float * v) const
{
	float dx = 1.f / (float)n;
	int i=0;
	float c = 0.f;
	for(;i<n;++i)
		c += dx * y[i];
		
	return c;
}

void Legendre2DTest::drawEvaluate(const float * y, GeoDrawer * dr)
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

void Legendre2DTest::drawLegendrePoly(int m, GeoDrawer * dr)
{
	glPushMatrix();
	glScaled(15.0, 10.0, 10.0);
	glBegin(GL_LINES);
	int i;
	for(i=0;i<M_NUM_EVAL-1;++i) {
		glVertex3f(INTERVAL_A + DX_SAMPLE * i, m_v[M_NUM_EVAL*m + i ], 0.f);
		glVertex3f(INTERVAL_A + DX_SAMPLE * (i+1), m_v[M_NUM_EVAL*m + i +1], 0.f);
	}
	glEnd();
	glPopMatrix();
}

}