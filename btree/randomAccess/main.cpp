/*
 *  main.cpp
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <stdint.h>
#include "../BTree.h"
#include "../Types.h"
#include "../Array.h"
#include "../../shared/PseudoNoise.h"
#include <boost/timer.hpp>
using namespace sdb;

struct Ptc {
	float vec[4];
	float pos[4];
};

void testUniform(const int & dim, const Ptc * pool)
{
	PseudoNoise noi;
	int coord;
	float sum = 0;
	boost::timer tm;
	for(int k=0; k < 159; k++)
	for(int j=0; j < 159; j++)
	for(int i=0; i < 159; i++) {
		coord = noi.rint2(i+1, j) % dim + (noi.rint2(i + 3, j) % dim) * dim + (noi.rint2(i, j + 7) % dim) * dim * dim;
		//std::cout<<" "<<noi.rint3(i, j, k);
		//coord = i + j * dim + k * dim * dim;
		//std::cout<<" "<<coord;
		sum += pool[coord].vec[0];
		sum += pool[coord].pos[1];
		sum += pool[coord].pos[2];
	}

	std::cout<<"sum "<<sum;
	std::cout<<"uniform test met "<<tm.elapsed();
}

int main()
{
	std::cout<<"b-tree test\ntry to insert a few keys\n";

	const int sptc = sizeof(Ptc);

	std::cout<<"ptc size "<<sptc<<std::endl;

	PseudoNoise noi;

	const int dim = 32;
	char *raw = new char[dim * dim * dim * sptc + 31];

	std::cout<<" "<<reinterpret_cast<uintptr_t>(raw);

	uintptr_t aligned = reinterpret_cast<uintptr_t>(raw+31)  & ~ 0x0F;

	Ptc *pool = (Ptc *)aligned;

	std::cout<<" aligned "<<reinterpret_cast<uintptr_t>(pool) % 32;

	for(int k=0; k < dim; k++)
		for(int j=0; j < dim; j++)
			for(int i=0; i < dim; i++) {
				pool->vec[0] = noi.rfloat(i);
				pool->pos[1] = noi.rfloat(j);
				pool->pos[2] = noi.rfloat(k);
				pool++;
			}
	pool -= dim*dim*dim;

	std::cout<<"filled pool\n";

	testUniform(dim, pool);

	std::cout<<"\ndone.\n";

	delete[] raw;
	return 0;
}
