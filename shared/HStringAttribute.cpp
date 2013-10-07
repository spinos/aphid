/*
 *  HStringAttribute.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "HStringAttribute.h"

HStringAttribute::HStringAttribute(const std::string & path) : HAttribute(path) {}
HStringAttribute::~HStringAttribute() {}
	
hid_t HStringAttribute::dataType()
{
	return H5T_NATIVE_CHAR;
}

char HStringAttribute::write(const std::string & str)
{
	herr_t status = H5Awrite(fObjectId, dataType(), str.c_str());
	if(status < 0)
		return 0;
	return 1;
}

char HStringAttribute::read(std::string & str)
{
	const int d = dataSpaceDimension();
	char * t = new char[d];
	herr_t status = H5Aread(fObjectId, dataType(), t);
	if(status < 0)
		return 0;
	str = std::string(t);
	delete[] t;
	return 1;
}