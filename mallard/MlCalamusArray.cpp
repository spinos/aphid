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
