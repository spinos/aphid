/*
 *  MlCalamusArray.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlCalamusArray.h"

MlCalamusArray::MlCalamusArray()
{
	setIndex(0);
	setElementSize(sizeof(MlCalamus));
	
	std::cout<<"size of calamus "<<sizeof(MlCalamus)<<"\n";
	std::cout<<"size of MlFeather * "<<sizeof(MlFeather *)<<"\n";
}

MlCalamusArray::~MlCalamusArray()
{
	clear();
}
	
MlCalamus * MlCalamusArray::asCalamus(unsigned index)
{
	return (MlCalamus *)at(index);
}

MlCalamus * MlCalamusArray::asCalamus(unsigned index) const
{
	return (MlCalamus *)at(index);
}

MlCalamus * MlCalamusArray::asCalamus()
{
	return (MlCalamus *)current();
}

float MlCalamusArray::sortKeyAt(unsigned idx) const
{
	MlCalamus * c = asCalamus(idx);
	return (float)c->faceIdx();
}

void MlCalamusArray::swapElement(unsigned a, unsigned b)
{
	MlCalamus t = *asCalamus(a);
	*asCalamus(a) = *asCalamus(b);
	*asCalamus(b) = t;
}

