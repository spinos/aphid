/*
 *  HStringAttribute.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "HStringAttribute.h"
#include <iostream>
namespace aphid {

HStringAttribute::HStringAttribute(const std::string & path) : HAttribute(path) {}
HStringAttribute::~HStringAttribute() {}
	
hid_t HStringAttribute::dataType()
{
	return H5T_NATIVE_CHAR;
}

char HStringAttribute::write(const std::string & str)
{
	herr_t status = H5Awrite(fObjectId, dataType(), str.c_str());
	if(status < 0)
		return 0;
	return 1;
}

char HStringAttribute::read(std::string & str)
{
	const int d = dataSpaceDimension();
	char * t = new char[d + 1];
	herr_t status = H5Aread(fObjectId, dataType(), t);
	if(status < 0)
		return 0;
	t[d] = '\0';
	str = std::string(t);
	delete[] t;
	return 1;
}

HVLStringAttribute::HVLStringAttribute(const std::string & path) : HStringAttribute(path) {}
HVLStringAttribute::~HVLStringAttribute() {}
	
hid_t HVLStringAttribute::dataType()
{
	hid_t type = H5Tcopy (H5T_C_S1);
	H5Tset_size (type, H5T_VARIABLE);
	return type;
}

char HVLStringAttribute::write(const std::string & str)
{
	const char *string_att[1];
	string_att[0] = str.c_str();
	herr_t status = H5Awrite(fObjectId, dataType(), &string_att);	
		
	if(status < 0)
		return 0;
	
	return 1;
}

char HVLStringAttribute::read(std::string & str)
{	
	hid_t ftype = H5Aget_type(fObjectId);
    H5T_class_t type_class = H5Tget_class (ftype);   

    if (type_class != H5T_STRING) {
		std::cout<<"\n type class is not H5T_STRING";
		return HStringAttribute::read(str);
	}
    
	char * t[1];
	herr_t status = H5Aread(fObjectId, dataType(), &t);
	if(status < 0)
		return 0;
	str = std::string(t[0]);
	free( t[0]);
	return 1;
}

}