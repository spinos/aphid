/*
 *  IndicesHDataset.cpp
 *  opium
 *
 *  Created by jian zhang on 6/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "IndicesHDataset.h"
#include <iostream>
IndicesHDataset::IndicesHDataset(const std::string & path) : HDataset(path)
{
}

IndicesHDataset::~IndicesHDataset() {}

void IndicesHDataset::setNumIndices(int num)
{
	fDimension[0] = num;
	fDimension[1] = 0;
	fDimension[2] = 0;
}
	
int IndicesHDataset::numIndices() const
{
	return fDimension[0];
}

hid_t IndicesHDataset::dataType()
{
	return H5T_NATIVE_INT;
}

