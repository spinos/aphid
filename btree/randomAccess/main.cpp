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

int main()
{
	std::cout<<"b-tree test\ntry to insert a few keys\n";

	PseudoNoise noi;

	const int dim = 32;
	char *raw = new char[dim * dim * dim * 4 + 31];

	std::cout<<" "<<reinterpret_cast<uintptr_t>(raw);

	uintptr_t aligned = reinterpret_cast<uintptr_t>(raw+31)  & ~ 0x0F;

	std::cout<<" "<<aligned;

	int *pool = (int *)aligned;
	for(int k=0; k < dim; k++)
		for(int j=0; j < dim; j++)
			for(int i=0; i < dim; i++) {
				*pool = noi.rint3(i, j, k) % 9999;
				pool++;
			}
	pool -= dim*dim*dim;

	std::cout<<"filled pool\n";

	int coord, sum = 0;
	boost::timer tm;
	for(int i=0; i < 1999; i++) {
		for(int j=0; j < 1999; j++) {
		coord = noi.rint2(i+1, j) % dim + (noi.rint2(i + 3, j) % dim) * dim + (noi.rint2(i, j + 7) % dim) * dim * dim;
		sum += pool[coord];
		}
	}

	std::cout<<"sum "<<sum;
	std::cout<<" met "<<tm.elapsed();
	std::cout<<"\ndone.\n";

	delete[] raw;
	return 0;
}
