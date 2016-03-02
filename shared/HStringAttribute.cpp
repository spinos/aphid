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

HVLStringAttribute::HVLStringAttribute(const std::string & path) : HAttribute(path) {}
HVLStringAttribute::~HVLStringAttribute() {}
	
hid_t HVLStringAttribute::dataType()
{
	hid_t type = H5Tcopy (H5T_C_S1);
	H5Tset_size (type, H5T_VARIABLE);
	return type;
}

char HVLStringAttribute::write(const std::string & str)
{
	hid_t ftype = H5Aget_type(fObjectId);

    H5T_class_t type_class = H5Tget_class (ftype);
	
	if (type_class == H5T_STRING) printf ("File datatype has class H5T_STRING\n");
    
	hsize_t size = H5Tget_size(ftype);

    printf(" Size is of the file datatype returned by H5Tget_size %d \n This is a size of char pointer\n Use H5Tis_variable_str call instead \n", size);
	
	htri_t size_var;
	if((size_var = H5Tis_variable_str(ftype)) == 1)
        printf(" to find if string has variable size \n");

	//hsize_t   dims[1] = {1};
	//hid_t dataSpace = H5Screate_simple(1, dims, NULL);
	
	hid_t type;// = //H5Tcopy (H5T_C_S1);
	//H5Tset_size (type, 5);
	type = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
    
	
	char string_att[] = "This";
		H5Awrite(fObjectId, dataType(), string_att);	
		
	//H5Sclose(dataSpace);
	return 1;
	herr_t status = H5Awrite(fObjectId, dataType(), str.c_str());
	if(status < 0)
		return 0;
	return 1;
}

char HVLStringAttribute::read(std::string & str)
{
	hid_t ftype = H5Aget_type(fObjectId);
	size_t size = H5Tget_size(ftype);
	char * t;
	herr_t status = H5Aread(fObjectId, dataType(), t);
	if(status < 0)
		return 0;
	str = std::string(t);
	free( t);
	return 1;
}

}