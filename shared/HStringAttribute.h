/*
 *  HStringAttribute.h
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <h5/HAttribute.h>

namespace aphid {

class HStringAttribute : public HAttribute
{
public:
	HStringAttribute(const std::string & path);
	~HStringAttribute();
	
	virtual hid_t dataType();
	virtual char write(const std::string & str);
	virtual char read(std::string & str);
};

/// variable length string attrib based on
/// http://www.hdfgroup.org/ftp/HDF5/examples/misc-examples/attrvstr.c
class HVLStringAttribute : public HStringAttribute
{
public:
	HVLStringAttribute(const std::string & path);
	~HVLStringAttribute();
	
	virtual hid_t dataType();
	virtual char write(const std::string & str);
	virtual char read(std::string & str);
};

}