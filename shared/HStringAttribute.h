/*
 *  HStringAttribute.h
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <HAttribute.h>

class HStringAttribute : public HAttribute
{
public:
	HStringAttribute(const std::string & path);
	~HStringAttribute();
	
	virtual hid_t dataType();
	virtual char write(const std::string & str);
	virtual char read(std::string & str);
};