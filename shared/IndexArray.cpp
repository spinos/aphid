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
	initialize();
}

IndexArray::~IndexArray() 
{
	clear();
}
/*
unsigned *IndexArray::asIndex(unsigned index)
{
	return (unsigned *)at(index);
}
*/
unsigned *IndexArray::asIndex(unsigned x) 
{
	return at(x);
}

unsigned *IndexArray::asIndex()
{
	return current();
}
//:~