/*
 *  XformHDataset.h
 *  opium
 *
 *  Created by jian zhang on 6/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <HDataset.h>
class XformHDataset : public HDataset {
public:
	XformHDataset(const std::string & path);
	virtual ~XformHDataset();
	virtual char read(float *data);
};