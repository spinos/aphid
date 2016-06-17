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
	int i = 0;
	for(;i<m_N;++i) {
		dr->cube(m_X[i], .125f);
	}
	
	dr->m_wireProfile.apply();
	dr->setColor(0.2f, 0.2f, 0.49f);
	
}

bool SuperformulaTest::createSamples()
{
	m_N = 4000;
	m_X = new Vector3F[m_N];
	std::cout<<"\n n sample "<<m_N;
	int i = 0;
	for(;i<m_N;++i) {
		m_X[i] = randomPnt(3.f, 3.f, 2.f, 1.f, 1.f, 1.f);
	}
	
	std::cout.flush();
	return true;
}

/// reference https://en.wikipedia.org/wiki/Superformula
Vector3F SuperformulaTest::randomPnt(float a, float b, float n1, float n2, float n3, float n4) const
{
	float u = RandomFn11() * 3.14159269f; /// longitude
	float v = RandomFn11() * 3.14159269f * .5f; /// latitude
	float raux1 = pow(Absolute<float>(1.f / a * Absolute<float>(cos(n1 * u / 4.f) ) ),
						n3)
				+ pow(Absolute<float>(1.f / b * Absolute<float>(sin(n1 * u / 4.f) ) ),
						n4);
	float r1 = pow(Absolute<float>(raux1), -1.f / n2 );
	float raux2 = pow(Absolute<float>(1.f / a * Absolute<float>(cos(n1 * v / 4.f) ) ),
						n3)
				+ pow(Absolute<float>(1.f / b * Absolute<float>(sin(n1 * v / 4.f) ) ),
						n4);
	float r2 = pow(Absolute<float>(raux2), -1.f / n2 );
					
	return Vector3F(r1 * cos(u) * r2 * cos(v),
					r1 * sin(u) * r2 * cos(v),
					r2 * sin(v) );
}

}
