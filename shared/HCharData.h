/*
 *  HCharData.h
 *  mallard
 *
 *  Created by jian zhang on 10/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <HDataset.h>
namespace aphid {

class HCharData : public HDataset {
public:
	HCharData(const std::string & path);
	virtual ~HCharData();
	
	void setNumChars(int num);
	int numChars() const;
	
	virtual hid_t dataType();
private:
};

}