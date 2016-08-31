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
#include <ANoise3.h>

using namespace aphid;
namespace ttg {

Legendre2DTest::Legendre2DTest() 
{}

Legendre2DTest::~Legendre2DTest() 
{}
	
const char * Legendre2DTest::titleStr() const
{ return "2D Legendre Polynomial Approximation"; }

bool Legendre2DTest::init()
{
	const float du = 1.f / 8;
	Vector3F smp, orp(.6241f, .4534f, .2786f);
	m_exact.create(16, 16);
	int i,j;
	for(j=0;j<=16;++j) {
		for(i=0;i<=16;++i) {
			smp.set(du*i - 1.f, 0.f, du*j - 1.f);
			smp.y = ANoise3::Fbm((const float *)&smp,
										(const float *)&orp,
										1.f,
										3,
										1.99f,
										.5f);
			*m_exact.quadP(i, j) = smp;
		}
	}
	
#define N_DIM 2
#define N_ORD 2

	int indx[N_DIM];
	
	int rnk = 0;
	int neval = 0;
	for(;;) {
		calc::tuple_next(1, N_ORD, N_DIM, &rnk, indx);
		
		if(rnk==0)
			break;
	
		calc::printValues<int>("tuple space", N_DIM, indx);
		
		neval++;
	}
	
	std::cout<<"\n n evaluate "<<neval
			<<"\n done!";
	std::cout.flush();
	return true;
}

void Legendre2DTest::draw(GeoDrawer * dr)
{
	glColor3f(0.f, 0.f, 0.f);
	glPushMatrix();
	glScalef(10.f, 10.f, 10.f);
	dr->m_wireProfile.apply();
	dr->geometry(&m_exact);
	glPopMatrix();
	glColor3f(0.33f, 1.f, 0.33f);
	
}

}