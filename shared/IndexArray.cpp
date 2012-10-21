/*
 *  IndexArray.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "IndexArray.h"

IndexArray::IndexArray()
{
	setIndex(0);
	setElementSize(sizeof(unsigned));
}

IndexArray::~IndexArray() 
{
	clear();
}

unsigned *IndexArray::asIndex(unsigned index)
{
	return (unsigned *)at(index);
}

unsigned *IndexArray::asIndex()
{
	return (unsigned *)current();
}