/*
 *  HIntAttribute.h
 *  helloHdf
 *
 *  Created by jian zhang on 12/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <HAttribute.h>

class HFloatAttribute : public HAttribute
{
public:
	HFloatAttribute(const std::string & path);
	~HFloatAttribute() {}
	virtual hid_t dataType();
	virtual char write(float *data);
	virtual char read(float *data);
};