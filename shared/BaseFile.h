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
	
	virtual bool create(const std::string & filename);
	virtual bool open(const std::string & filename);
	virtual bool open();
	virtual bool save();
	virtual bool close();
	
	void setFileName(const std::string & filename);
	std::string fileName() const;
	
	void setOpened();
	void setClosed();
	bool isOpened() const;
	bool isUntitled() const;
	
	void setLatestError(BaseFile::ErrorMsg err);
	BaseFile::ErrorMsg latestError() const;
	
	static bool FileExists(const std::string & name);
	
private:
	std::string m_fileName;
	ErrorMsg m_error;
	bool _opened;
};