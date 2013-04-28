/*
 *  HFloatAttribute.cpp
 *  helloHdf
 *
 *  Created by jian zhang on 12/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "HFloatAttribute.h"

HFloatAttribute::HFloatAttribute(const std::string & path)
{
	fObjectPath = ValidPathName(path);
}

hid_t HFloatAttribute::dataType()
{
	return H5T_NATIVE_FLOAT;
}

char HFloatAttribute::write(float *data)
{
	herr_t status = H5Awrite(fObjectId, dataType(), data);
	if(status < 0)
		return 0;
	return 1;
}

char HFloatAttribute::read(float *data)
{
	herr_t status = H5Aread(fObjectId, dataType(), data);
	if(status < 0)
		return 0;
	return 1;
}