/*
 *  Noise3Test.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Noise3Test.h"
#include <NTreeDrawer.h>
#include <ANoise3.h>
#include <iostream>

using namespace aphid;
namespace ttg {

Noise3Test::Noise3Test() 
{}

Noise3Test::~Noise3Test() 
{}
	
const char * Noise3Test::titleStr() const
{ return "Noise 3D Test"; }

bool Noise3Test::init()
{ return true; }

void Noise3Test::draw(GeoDrawer * dr)
{
	Vector3F o(0.435f, 0.0036f, 0.765f);
#define d 3.5f
#define Dim 19
#define HDim 5
	int i, j, k;
	for(k=0;k<Dim;++k) {
		for(j=0;j<Dim;++j) {
			for(i=0;i<Dim;++i) {
				
				Vector3F p(d*(i - HDim), d*(j - HDim), d*(k - HDim) );
				
				float r = ANoise3::FractalF((const float *)&p,
											(const float *)&o,
											.125f,
											1.5f,
											4);
										
				dr->setColor(0.f, r, 0.f);
				dr->cube(p, 1.f);
			}
		}
	}
}

}