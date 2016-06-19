/*
 *  SuperformulaPoisson.cpp
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "SuperformulaPoisson.h"
#include <iostream>

using namespace aphid;
namespace ttg {

SuperformulaPoisson::SuperformulaPoisson()
{}

SuperformulaPoisson::~SuperformulaPoisson() 
{}

const char * SuperformulaPoisson::titleStr() const
{ return "Superformula / Poisson-disk Sampling"; }

bool SuperformulaPoisson::createSamples()
{
	BoundingBox box;
	int i, j;
	const float du = 2.f * 3.14159269f / 120.f;
	const float dv = 3.14159269f / 60.f;
	
	for(j=0;j<60;++j) {
		for(i=0;i<120;++i) {
			box.expandBy(randomPnt(du * i - 3.1415927f, dv * j - 1.507963f) );
		}
	}

/// 1 / 32 of size of the boundary
	float r = box.getLongestDistance() * .03125f;
	float gridSize = r * 3.f;
	
	PoissonSequence<Disk> bkg;
	bkg.setGridSize(gridSize);
	
/// limit of n
	setN(3000);
	
	int numAccept = 0, preN = 0;
	Disk cand;
	cand.r = r * .5f;
/// 25 times of n
	for(i=0; i<75000; ++i) {
	
		cand.pos = randomPnt(RandomFn11() * 3.14159269f, 
								RandomFn11() * 3.14159269f * .5f);
		
		if(!bkg.reject(&cand) ) {
			
			Disk * samp = new Disk;
			samp->pos = cand.pos;
			samp->r = cand.r;
			samp->key = numAccept;
			
			bkg.insert((const float *)&cand.pos, samp);
			
			X()[numAccept++] = cand.pos;
		}
		
		if((i+1) & 7 == 0) {
			if(numAccept == preN) break;
			preN = numAccept;
		}
		if(numAccept >= 3000) break;
	}
	
	setNDraw(numAccept);
	bkg.clear();
	std::cout<<"\n n accept "<<numAccept;
	
	std::cout.flush();
	return true;
}

}
