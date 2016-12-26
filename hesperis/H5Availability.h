/*
 *  H5Availability.h
 *  opium
 *
 *  Created by jian zhang on 6/19/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_H5_AVAILABILITY_H
#define APH_H5_AVAILABILITY_H
#include <h5/HDocument.h>
#include <string>
#include <map>
namespace aphid {
    
class H5Availability {
public:
	H5Availability();
	~H5Availability();
	
	char openFile(const char* filename, HDocument::OpenMode accessFlag);
	char closeFile(const char* filename);
	static std::string h5FileName(const char* filename);
	std::string closeAll();
	
private:
	std::map<std::string, HDocument * > fFileStatus;
};

}
#endif