/*
 *  HIntAttribute.cpp
 *  helloHdf
 *
 *  Created by jian zhang on 12/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "HIntAttribute.h"

HIntAttribute::HIntAttribute(const std::string & path) : HAttribute(path)
{
}

char HIntAttribute::write(int *data)
{
	herr_t status = H5Awrite(fObjectId, dataType(), data);
	if(status < 0)
		return 0;
	return 1;
}

char HIntAttribute::read(int *data)
{
	herr_t status = H5Aread(fObjectId, dataType(), data);
	if(status < 0)
		return 0;
	return 1;
}