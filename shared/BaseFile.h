/*
 *  BaseFile.h
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <string>

class BaseFile {
public:
	BaseFile();
	BaseFile(const char * name);
	std::string fileName() const;
	static bool FileExists(const std::string & name);
	
	std::string m_fileName;
};