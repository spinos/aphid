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
	enum ErrorMsg {
		NoError = 0,
		FileNotFound = 1,
		FileNotWritable = 2,
		FileNotReadable = 3
	};
	
	BaseFile();
	BaseFile(const char * name);
	virtual char load(const char * filename);
	std::string fileName() const;
	char isValid() const;
	
	void setLatestError(BaseFile::ErrorMsg err);
	BaseFile::ErrorMsg latestError() const;
	
	static bool FileExists(const std::string & name);
	
	std::string m_fileName;
	char _valid;
private:
	ErrorMsg m_error;
};