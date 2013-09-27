#ifndef HDATASET_H
#define HDATASET_H

/*
 *  HDataset.h
 *  helloHdf
 *
 *  Created by jian zhang on 6/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "HObject.h"
class HDataset : public HObject {
public:
	HDataset() {}
	HDataset(const std::string & path);
	virtual ~HDataset() {}
	
	virtual char validate();
	virtual char create(int dimx, int dimy);
	virtual char raw_create();

	virtual char open();
	virtual void close();
	virtual int objectType() const;
	
	virtual hid_t dataType();
	virtual void createDataSpace();
	virtual char verifyDataSpace();
	virtual char dimensionMatched();
	
	virtual char write();
	virtual char read();
	virtual char write(float *data);
	virtual char read(float *data);
	
	int dataSpaceNumDimensions() const;
	void dataSpaceDimensions(int dim[3]) const;
	
	hid_t fDataSpace;
	int fDimension[3];
	
};
#endif        //  #ifndef HDATASET_H
