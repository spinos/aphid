/*
 *  IndicesHDataset.h
 *  opium
 *
 *  Created by jian zhang on 6/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <HDataset.h>

class IndicesHDataset : public HDataset {
public:
	IndicesHDataset(const std::string & path);
	virtual ~IndicesHDataset();
	
	void setNumIndices(int num);
	int numIndices() const;
	
	virtual hid_t dataType();
	
	virtual char create();
	virtual char write(int *data);
	virtual char read(int *data);
};