/*
 *  HAttribute.cpp
 *  helloHdf
 *
 *  Created by jian zhang on 12/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "HAttribute.h"

HAttribute::HAttribute(const std::string & path) : HObject(path)
{
}

char HAttribute::create(int dim, hid_t parentId)
{
	hsize_t dims = dim;
	
	hid_t dataSpace = H5Screate_simple(1, &dims, NULL);
	
	fObjectId = H5Acreate(parentId, fObjectPath.c_str(), dataType(), dataSpace, 
                          H5P_DEFAULT, H5P_DEFAULT);
						  
	H5Sclose(dataSpace);
	if(fObjectId < 0)
		return 0;
	return 1;
}

char HAttribute::open(hid_t parentId)
{
	fObjectId = H5Aopen(parentId, fObjectPath.c_str(), H5P_DEFAULT);
	
	if(fObjectId < 0)
		return 0;
		
	return 1;
}

void HAttribute::close()
{
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
	hid_t fDataSpace = H5Aget_space(fObjectId);
	hsize_t     dims_out[3];
	H5Sget_simple_extent_dims(fDataSpace, dims_out, NULL);
	return dims_out[0];
}
//:~