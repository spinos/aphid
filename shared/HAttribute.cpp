/*
 *  HAttribute.cpp
 *  helloHdf
 *
 *  Created by jian zhang on 12/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "HAttribute.h"

HAttribute::HAttribute(const std::string & path)
{
	fObjectPath = ValidPathName(path);
}

char HAttribute::create(int dim, hid_t parentId)
{
	hsize_t dims = dim;
	
	fDataSpace = H5Screate_simple(1, &dims, NULL);
	
	fObjectId = H5Acreate(parentId, fObjectPath.c_str(), dataType(), fDataSpace, 
                          H5P_DEFAULT, H5P_DEFAULT);
						  
	if(fObjectId < 0)
		return 0;
	return 1;
}

char HAttribute::open(hid_t parentId)
{
	fObjectId = H5Aopen(parentId, fObjectPath.c_str(), H5P_DEFAULT);
	
	if(fObjectId < 0)
		return 0;
		
	fDataSpace = H5Aget_space(fObjectId);

	if(fDataSpace<0)
		return 0;
		
	return 1;
}

void HAttribute::close()
{
	H5Sclose(fDataSpace);
	H5Aclose(fObjectId);
}

int HAttribute::objectType() const
{
	return H5T_STD_I32BE;
}

hid_t HAttribute::dataType()
{
	return H5T_NATIVE_INT;
}

int HAttribute::dataSpaceDimension() const
{
	hsize_t     dims_out[3];
	H5Sget_simple_extent_dims(fDataSpace, dims_out, NULL);
	return dims_out[0];
}
//:~