/*
 *  XformHDataset.h
 *  opium
 *
 *  Created by jian zhang on 6/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_XFORM_H_DATA_SET_H
#define APH_XFORM_H_DATA_SET_H
#include <h5/HDataset.h>
namespace aphid {

class XformHDataset : public HDataset {
public:
	XformHDataset(const std::string & path);
	virtual ~XformHDataset();
	virtual char read(float *data);
};

}
#endif