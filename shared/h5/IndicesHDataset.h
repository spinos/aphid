/*
 *  IndicesHDataset.h
 *  opium
 *
 *  Created by jian zhang on 6/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_INDICES_H_DATA_SET_H
#define APH_INDICES_H_DATA_SET_H
#include <h5/HDataset.h>
namespace aphid {

class IndicesHDataset : public HDataset {
public:
	IndicesHDataset(const std::string & path);
	virtual ~IndicesHDataset();
	
	void setNumIndices(int num);
	int numIndices() const;
	
	virtual hid_t dataType();
};

}
#endif