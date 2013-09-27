/*
 *  HGroup.cpp
 *  helloHdf
 *
 *  Created by jian zhang on 6/12/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "HGroup.h"

HGroup::HGroup(const std::string & path) : HObject(path)
{
}

char HGroup::create()
{
	if(validate())
		return 1;
		
	fObjectId = H5Gcreate(FileIO.fFileId, fObjectPath.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(fObjectId < 0)
		return 0;
		
	return 1;
}

char HGroup::open()
{
	fObjectId = H5Gopen(FileIO.fFileId, fObjectPath.c_str(), H5P_DEFAULT);
	
	if(fObjectId < 0)
		return 0;

	return 1;
}

void HGroup::close()
{
	H5Gclose(fObjectId);
}

int HGroup::objectType() const
{
	return H5G_GROUP;
}
