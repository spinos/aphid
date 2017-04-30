#ifndef APH_H_DATASET_H
#define APH_H_DATASET_H

/*
 *  HDataset.h
 *  helloHdf
 *
 *  Created by jian zhang on 6/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "HObject.h"

namespace aphid {

class HDataset : public HObject {
public:
	struct SelectPart {
		SelectPart() {
			start[0] = 0;
			stride[0] = 1;
			count[0] = 1;
			block[0] = 1;
		}
		
		hsize_t start[1];
		hsize_t stride[1];
		hsize_t count[1];
		hsize_t block[1];
	};
	
	HDataset(const std::string & path);
	virtual ~HDataset() {}
	
	char create(hid_t parentId);

	virtual char open(hid_t parentId = FileIO.fFileId);
	virtual void close();
	virtual int objectType() const;
	
	virtual hid_t dataType();

	virtual char write(char *data, SelectPart * part = 0);
	virtual char read(char *data, SelectPart * part = 0);
	
	void getSpaceDimension(int * dimensions, int * ndimension) const;
	char hasEnoughSpace() const;
	void resize();
	
	hsize_t fDimension[3];
	
private:
	hid_t createMemSpace() const;
	hid_t createFileSpace() const;
	hid_t createSpace(hsize_t * block) const;
};

}
#endif        //  #ifndef HDATASET_H
