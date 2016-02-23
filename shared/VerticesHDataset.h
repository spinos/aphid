/*
 *  VerticesHDataset.h
 *  opium
 *
 *  Created by jian zhang on 6/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <HDataset.h>
namespace aphid {

class VerticesHDataset : public HDataset {
public:
	VerticesHDataset(const std::string & path);
	virtual ~VerticesHDataset();
	
	void setNumVertices(int num);
	int numVertices() const;
};

}