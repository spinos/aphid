/*
 *  FloatsHDataset.h
 *  mallard
 *
 *  Created by jian zhang on 9/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <HDataset.h>

class FloatsHDataset : public HDataset {
public:
	FloatsHDataset(const std::string & path);
	virtual ~FloatsHDataset();
	
	void setNumFloats(int num);
	int numFloats() const;
	
	virtual char create(hid_t parentId);
};