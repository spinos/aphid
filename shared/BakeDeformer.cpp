/*
 *  BakeDeformer.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BakeDeformer.h"
#include <AllHdf.h>
BakeDeformer::BakeDeformer() {}
BakeDeformer::~BakeDeformer() {}

char BakeDeformer::load(const char * filename)
{
	if(!HObject::FileIO.open(filename, HDocument::oReadOnly)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	std::cout<<"read bake from "<<filename<<"\n";
	
	HObject::FileIO.close();
	return BaseFile::load(filename);
}