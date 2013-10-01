/*
 *  HCharData.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "HCharData.h"

HCharData::HCharData(const std::string & path)  : HDataset(path) {}
HCharData::~HCharData() {}
	
void HCharData::setNumChars(int num)
{
	fDimension[0] = num;
	fDimension[1] = 0;
	fDimension[2] = 0;
}

int HCharData::numChars() const
{
	return fDimension[0];
}
	
hid_t HCharData::dataType()
{
	return H5T_NATIVE_CHAR;
}