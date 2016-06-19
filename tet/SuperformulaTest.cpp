/*
 *  SuperformulaTest.cpp
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "SuperformulaTest.h"
#include <iostream>

using namespace aphid;
namespace ttg {

SuperformulaBase::SuperformulaBase() :
m_a1(1.f), m_b1(1.f), m_m1(4.f), m_n1(10.f), m_n2(10.f), m_n3(10.f),
m_a2(1.f), m_b2(1.f), m_m2(4.f), m_n21(10.f), m_n22(10.f), m_n23(10.f)
{}

SuperformulaBase::~SuperformulaBase()
{}

bool SuperformulaBase::createSamples()
{ return true; }

void SuperformulaBase::setA1(double x)
{
	m_a1 = x;
	createSamples();
}

void SuperformulaBase::setB1(double x)
{
	m_b1 = x;
	createSamples();
}

void SuperformulaBase::setM1(double x)
{
	m_m1 = x;
	createSamples();
}

void SuperformulaBase::setN1(double x)
{
	m_n1 = x;
	createSamples();
}

void SuperformulaBase::setN2(double x)
{
	m_n2 = x;
	createSamples();
}

void SuperformulaBase::setN3(double x)
{
	m_n3 = x;
	createSamples();
}

void SuperformulaBase::setA2(double x)
{
	m_a2 = x;
	createSamples();
}

void SuperformulaBase::setB2(double x)
{
	m_b2 = x;
	createSamples();
}

void SuperformulaBase::setM2(double x)
{
	m_m2 = x;
	createSamples();
}

void SuperformulaBase::setN21(double x)
{
	m_n21 = x;
	createSamples();
}

void SuperformulaBase::setN22(double x)
{
	m_n22 = x;
	createSamples();
}

void SuperformulaBase::setN23(double x)
{
	m_n23 = x;
	createSamples();
}

Vector3F SuperformulaBase::randomPnt(float u, float v) const
{ 
	return randomPnt(u, v, 
			m_a1, m_b1,
			m_m1, m_n1, m_n2, m_n3,
			m_a2, m_b2,
			m_m2, m_n21, m_n22, m_n23); 
}

/// reference https://en.wikipedia.org/wiki/Superformula
/// http://paulbourke.net/geometry/supershape/
Vector3F SuperformulaBase::randomPnt(float u, float v, float a, float b, 
									float m, float n1, float n2, float n3,
									float a2, float b2,
									float m2, float n21, float n22, float n23) const
{
//	float u = RandomFn11() * 3.14159269f; /// longitude
//	float v = RandomFn11() * 3.14159269f * .5f; /// latitude
	float raux1 = pow(Absolute<float>(1.f / a * Absolute<float>(cos(m * u / 4.f) ) ),
						n2)
				+ pow(Absolute<float>(1.f / b * Absolute<float>(sin(m * u / 4.f) ) ),
						n3);
	float r1 = pow(Absolute<float>(raux1), -1.f / n1 );
	float raux2 = pow(Absolute<float>(1.f / a2 * Absolute<float>(cos(m2 * v / 4.f) ) ),
						n22)
				+ pow(Absolute<float>(1.f / b2 * Absolute<float>(sin(m2 * v / 4.f) ) ),
						n23);
	float r2 = pow(Absolute<float>(raux2), -1.f / n21 );
					
	return Vector3F(r1 * cos(u) * r2 * cos(v),
					r1 * sin(u) * r2 * cos(v),
					r2 * sin(v) );
}

SuperformulaTest::SuperformulaTest() :
m_X(NULL)
{}

SuperformulaTest::~SuperformulaTest() 
{
	if(m_X) delete[] m_X;
}

bool SuperformulaTest::init() 
{ 
    return createSamples();
}

bool SuperformulaTest::progressForward()
{ 
	return true; 
}

bool SuperformulaTest::progressBackward()
{ 
	return true; 
}

const char * SuperformulaTest::titleStr() const
{ return "Superformula Test"; }

void SuperformulaTest::draw(GeoDrawer * dr) 
{
	dr->setColor(0.3f, 0.3f, 0.39f);
	
	dr->m_markerProfile.apply();
	dr->setColor(0.f, 0.f, 0.f);
	glBegin(GL_POINTS);
	int i = 0;
	for(;i<m_NDraw;++i) {
		// dr->cube(m_X[i], .025f);
		glVertex3fv((const float *)&m_X[i] );
	}
	glEnd();
	dr->m_wireProfile.apply();
	dr->setColor(0.2f, 0.2f, 0.49f);
	
}

void SuperformulaTest::setN(int x)
{
	m_N = x;
	if(m_X) delete[] m_X;
	m_X = new Vector3F[m_N];
	m_NDraw = x;
}

void SuperformulaTest::setNDraw(int x)
{ m_NDraw = x; }

aphid::Vector3F * SuperformulaTest::X()
{ return m_X; }

bool SuperformulaTest::createSamples()
{
	setN(7200);
	float du = 2.f * 3.14159269f / 120.f;
	float dv = 3.14159269f / 60.f;
	int i, j;
	for(j=0;j<60;++j) {
		for(i=0;i<120;++i) {
			m_X[j*120+i] = randomPnt(du * i - 3.1415927f, dv * j - 1.507963f);
		}
	}
	
	std::cout.flush();
	return true;
}

}
