/*
 *  IndicesHDataset.cpp
 *  opium
 *
 *  Created by jian zhang on 6/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "IndicesHDataset.h"

IndicesHDataset::IndicesHDataset() {}

IndicesHDataset::IndicesHDataset(const std::string & path)
{
	fObjectPath = ValidPathName(path);
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

char IndicesHDataset::create()
{
	return raw_create();
}

char IndicesHDataset::write(int *data)
{
	herr_t status = H5Dwrite(fObjectId, dataType(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if(status < 0)
		return 0;
	return 1;
}

char IndicesHDataset::read(int *data)
{
	herr_t status = H5Dread(fObjectId, dataType(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if(status < 0)
		return 0;
	return 1;
}
