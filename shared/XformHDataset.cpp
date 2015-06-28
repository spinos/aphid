/*
 *  XformHDataset.cpp
 *  opium
 *
 *  Created by jian zhang on 6/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "XformHDataset.h"

XformHDataset::XformHDataset(const std::string & path) : HDataset(path)
{
    fDimension[0] = 4;
	fDimension[1] = 4;
	fDimension[2] = 0;
}

XformHDataset::~XformHDataset() {}

