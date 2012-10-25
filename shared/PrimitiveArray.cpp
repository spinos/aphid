/*
 *  PrimitiveArray.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "PrimitiveArray.h"

PrimitiveArray::PrimitiveArray() 
{
	setIndex(0);
	setElementSize(sizeof(Primitive));
}

PrimitiveArray::~PrimitiveArray() 
{
	clear();
}

Primitive *PrimitiveArray::asPrimitive(unsigned index)
{
	return (Primitive *)at(index);
}

Primitive *PrimitiveArray::asPrimitive(unsigned index) const
{
	return (Primitive *)at(index);
}

Primitive *PrimitiveArray::asPrimitive()
{
	return (Primitive *)current();
}
