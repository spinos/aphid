/*
 *  HAttribute.h
 *  helloHdf
 *
 *  Created by jian zhang on 12/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "HObject.h"

namespace aphid {

class HAttribute : public HObject {
public:
	HAttribute(const std::string & path);
	virtual ~HAttribute() {}
	
	virtual char create(int dim, hid_t parentId = FileIO.fFileId);

	virtual char open(hid_t parentId = FileIO.fFileId);
	virtual void close();
	virtual int objectType() const;
	virtual hid_t dataType();
	
	int dataSpaceDimension() const;
};

}