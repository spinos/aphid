/*
 *  H5Availability.h
 *  opium
 *
 *  Created by jian zhang on 6/19/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <HDocument.h>
#include <string>
#include <map>
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