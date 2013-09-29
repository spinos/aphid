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
	HDataset(const std::string & path);
	virtual ~HDataset() {}
	
	char create(hid_t parentId);

	virtual char open(hid_t parentId = FileIO.fFileId);
	virtual void close();
	virtual int objectType() const;
	
	virtual hid_t dataType();
	
	char hasEnoughSpace() const;
	
	virtual char write(char *data);
	virtual char read(char *data);
	
	hid_t createMemDataSpace() const;
	hid_t createFileDataSpace() const;
	
	void resize();
	
	hsize_t fDimension[3];

private:
	
};
#endif        //  #ifndef HDATASET_H
