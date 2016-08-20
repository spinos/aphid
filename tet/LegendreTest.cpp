/*
 *  LegendreTest.cpp
 *  foo
 *
 *  http://mathfaculty.fullerton.edu/mathews/n2003/LegendrePolyMod.html
 *  http://faculty.washington.edu/finlayso/ebook/bvp/OC/Legendre.htm
 *  http://www.mhtlab.uwaterloo.ca/courses/me755/web_chap5.pdf
 *  http://archive.lib.msu.edu/crcmath/math/math/l/l175.htm
 *  http://web.cs.iastate.edu/~cs577/handouts/orthogonal-polys.pdf
 *  
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "LegendreTest.h"
#include <GridTables.h>
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

bool LegendreTest::init()
{
	int i;
	for(i=0;i<NUM_SAMPLE;++i) {
		m_exactEvaluate[i] = evaluateExact(DX_SAMPLE * i);
	}
	
	computeLegendrePoly();
	
	int m = 2, n = 5;
	std::cout<<"\n deg "<<m<<" order "<<n;
	
	equalSpacingNodes(n, m_x);
	
	for(i=0;i<n;++i) {
		m_y[i] = evaluateExact(m_x[i]);
	}
	
	printValues("X", n, m_x);
	printValues("Y", n, m_y);
	
	//legendreValue(m, n, m_x, m_v);
	//computePipi(m_pipi, m, n, m_v);
	
	computePix(m_v, m_pipi, m_xpipi, m, n, m_x);
	computeCoeff(m_coeff, m, n, m_y, m_v, m_pipi);
	computeApproximated(m_approximated, m, m_coeff);
	
	std::cout<<"\n done!";
	std::cout.flush();
	return true;
}

void LegendreTest::computePix(float * m_v, float * pipi, float * xpipi,
					int m, int n, const float * x) const
{
	int i, k;
	for(k=0; k<=m;++k) {
		pipi[k] = xpipi[k] = 0.f;
	}
	
	float pm1 = 0.f, pm2 = 0.f;
	for(k=0; k<=m;++k) {
		for(i=0; i<n; ++i) {
			if(k>1) {
				pm1 = m_v[i*(m+1) + k - 1];
				pm2 = m_v[i*(m+1) + k - 2];
			}
			
			m_v[i*(m+1) + k] = computePi(k, pipi, xpipi, x[i], pm1, pm2);
			
			std::cout<<"\n p"<<k<<"(x)["<<i<<"] "<<m_v[i*(m+1) + k];
			
		}
		
		pipi[k] = integratePipi(m_v, k, m, n);
		xpipi[k] = integrateXpipi(m_v, m_x, k, m, n);
		
		std::cout<<"\n <p"<<k<<",p"<<k<<"> "<<pipi[k]
				<<"\n <xp"<<k<<",p"<<k<<"> "<<xpipi[k];
		
	}
}

float LegendreTest::computePi(int k, const float * pipi, const float * xpipi, const float & x,
								const float & pm1, const float & pm2) const
{
	if(k==0) 
		return 1.f;
		
	if(k==1)
		return (x - xpipi[0] / pipi[0]);
/// orthogonal polynomial sequence		
	return (x - xpipi[k-1] / pipi[k-1] ) * pm1 - pipi[k-1] / pipi[k-2] * pm2;
	
}

float LegendreTest::integratePipi(const float * v, int k, int m, int n) const
{
	float c = 0.f;
	int i = 0;
	for(;i<n;++i)
		c += v[i*(m+1) + k] * v[i*(m+1) + k];
		
	return c;
}

float LegendreTest::integrateXpipi(const float * v, const float * x, int k, int m, int n) const
{
	float c = 0.f;
	int i = 0;
	for(;i<n;++i)
		c += x[i] * v[i*(m+1) + k] * v[i*(m+1) + k];
		
	return c;
}

void LegendreTest::equalSpacingNodes(int n, float * x) const
{
	float dx = 1.f / (float)(n-1);
	int i=0;
	for(;i<n;++i)
		x[i] = dx * i;
}

void LegendreTest::computeLegendrePoly()
{
	float pj0 = 0.f, pj1 = 0.f;
	int i, j;
	
	for(i=0;i<NUM_SAMPLE;++i) {
		for(j=0; j<=POLY_MAX_DEG; ++j) {
			if(j>1) {
				pj0 = m_ps[j-2][i];
				pj1 = m_ps[j-1][i];
			}
			m_ps[j][i] = legendreValue1(j, DX_SAMPLE * i, pj1, pj0);
		}
	}
}

float LegendreTest::legendreValue1(int m, const float & x,
									const float & pxm1, const float & pxm2) const
{
	if(m==0)
		return 1.f;
		
	if(m==1)
		return 2.0 * x - 1.0;
		
	return ( ( float ) ( 2 * m - 1 ) * ( 2.0 * x - 1.0 ) * pxm1   
                 - ( float ) (     m - 1 ) *              pxm2 ) 
                 / ( float ) (     m     );
}

void LegendreTest::legendreValue(int m, int n, const float * x, float * v) const
{
	float pj0 = 0.f, pj1 = 0.f;
	int i, j;
	for(i=0;i<n;++i) {
		for(j=0; j<=m; ++j) {
			if(j>1) {
				pj0 = m_ps[j-2][i];
				pj1 = m_ps[j-1][i];
			}
			v[i*(m+1) + j] = legendreValue1(j, x[i], pj1, pj0);
			std::cout<<"\n "<<j<<" "<<i<<" "<<v[i*(m+1) + j];
		}
	}
/*	
/// l 0
	for ( i = 0; i < m; i++ ) {
		v[i+0*m] = 1.0;
	}
/// l 1
	for ( i = 0; i < m; i++ ) {
		v[i+1*m] = 2.0 * x[i] - 1.0;
	}
/// l 2, 3, ...	
	for ( j = 2; j <= n; j++ ) {
		for ( i = 0; i < m; i++ ) {
			v[i+j*m] = ( ( float ) ( 2 * j - 1 ) * ( 2.0 * x[i] - 1.0 ) * v[i+(j-1)*m]   
                 - ( float ) (     j - 1 ) *                        v[i+(j-2)*m] ) 
                 / ( float ) (     j     );
		}
	}*/
}

void LegendreTest::computePipi(float * pipi, int m, int n, const float * v) const
{
	int i, j;
	for(i=0; i<=m; ++i) {
		pipi[i] = 0.f;
		for(j=0; j<n; ++j) {
			std::cout<<"\n "<<i<<" "<<j<<" "<<v[j*(m+1)+i];
			pipi[i] += v[j*(m+1)+i] * v[j*(m+1)+i];
		}
			
		std::cout<<"\n <p"<<i<<",p"<<i<<"> "<<pipi[i];
	}
}

void LegendreTest::computeCoeff(float * coeff, int m, int n, const float * y, const float * v, const float * pipi) const
{
	int i, j;
	for(i=0; i<=m; ++i) {
		coeff[i] = 0.f;
		for(j=0; j<n; ++j) {
			coeff[i] += y[j] * v[j*(m+1)+i];
		}
		
		coeff[i] /= pipi[i];
		std::cout<<"\n c"<<i<<" "<<coeff[i];
	}
}

void LegendreTest::computeApproximated(float * yhat, int m, const float * coeff) const
{
	float x, xp;
	int i, j;
	for(i=0;i<NUM_SAMPLE;++i) {
		x = (DX_SAMPLE * i - 0.5f);
		xp = 1.f;
		
		yhat[i] = 0.f;
		for(j=0; j<=m; ++j) {
			yhat[i] += coeff[j] * xp;
			
			xp *= x;
		}
	}
}

void LegendreTest::draw(GeoDrawer * dr)
{
	glColor3f(0.f, 0.f, 0.f);
	drawEvaluate(m_exactEvaluate, dr);
	
	glColor3f(0.33f, 1.f, 0.33f);
	drawEvaluate(m_approximated, dr);

#if 1
	int i=0;
	Vector3F mcol;
	for(;i<=POLY_MAX_DEG;++i) {
		sdb::gdt::GetCellColor(mcol, i);
		glColor3fv((const float *)&mcol);
		drawLegendrePoly(i, dr);
	}
#endif
}

float LegendreTest::evaluateExact(const float & x) const
{ 
	return 2.f - x* .1f - x * x / 3.f;
//	return exp(x * .845f) - .34f * sqrt(x); 
}

void LegendreTest::printValues(const char * h, int n, const float * y) const
{
	std::cout<<"\n "<<h;
	int i=0;
	for(;i<n;++i)
		std::cout<<" "<<std::setw(10) <<std::setprecision(8)<<y[i];
}

void LegendreTest::printLegendreValues(int m, int n, const float * x, float * v) const
{
	int i, j;
	std::cout<<"\n l0";
	for ( i = 0; i < m; i++ ) {
		std::cout<<" "<<std::setw(10) <<std::setprecision(8)<<v[i+0*m];
	}
	std::cout<<"\n l1";
	for ( i = 0; i < m; i++ ) {
		std::cout<<" "<<std::setw(10) <<std::setprecision(8)<<v[i+1*m];
	}	
	for ( j = 2; j <= n; j++ ) {
	std::cout<<"\n l"<<j;
		for ( i = 0; i < m; i++ ) {
			std::cout<<" "<<std::setw(10) <<std::setprecision(8)<<v[i+j*m];
		}
	}
}

void LegendreTest::findCoefficients(int m, int n, const float * y, const float * v, float * ci) const
{
	
}

float LegendreTest::discreteIntegral(int n, const float * y, const float * v) const
{
	float dx = 1.f / (float)n;
	int i=0;
	float c = 0.f;
	for(;i<n;++i)
		c += dx * y[i];
		
	return c;
}

void LegendreTest::drawEvaluate(const float * y, GeoDrawer * dr)
{
	glPushMatrix();
	glScaled(20.0, 10.0, 10.0);
	glBegin(GL_LINES);
	int i;
	for(i=0;i<NUM_SAMPLE-1;++i) {
		glVertex3f(DX_SAMPLE * i, y[i], 0.f);
		glVertex3f(DX_SAMPLE * (i+1), y[i+1], 0.f);
	}
	glEnd();
	glPopMatrix();
}

void LegendreTest::drawLegendrePoly(int m, GeoDrawer * dr)
{
	glPushMatrix();
	glScaled(20.0, 10.0, 10.0);
	glBegin(GL_LINES);
	int i;
	for(i=0;i<NUM_SAMPLE-1;++i) {
		glVertex3f(DX_SAMPLE * i, m_ps[m][i], 0.f);
		glVertex3f(DX_SAMPLE * (i+1), m_ps[m][i+1], 0.f);
	}
	glEnd();
	glPopMatrix();
}

}