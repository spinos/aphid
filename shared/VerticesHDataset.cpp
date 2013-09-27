/*
 *  VerticesHDataset.cpp
 *  opium
 *
 *  Created by jian zhang on 6/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "VerticesHDataset.h"

VerticesHDataset::VerticesHDataset(const std::string & path) : HDataset(path)
{
}

VerticesHDataset::~VerticesHDataset() {}

void VerticesHDataset::setNumVertices(int num)
{
	fDimension[0] = num * 3;
	fDimension[1] = 0;
	fDimension[2] = 0;
}
	
int VerticesHDataset::numVertices() const
{
	return fDimension[0] / 3;
}

char VerticesHDataset::create(hid_t parentId)
{
	return raw_create(parentId);
}