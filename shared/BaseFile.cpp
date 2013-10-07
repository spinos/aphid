/*
 *  BaseFile.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseFile.h"
#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>
BaseFile::BaseFile() 
{
	m_fileName = "unknown";
	_valid = 0;
}

BaseFile::BaseFile(const char * name)
{
	m_fileName = std::string(name);
	_valid = 0;
}

char BaseFile::load(const char *filename)
{
	m_fileName = filename;
	_valid = 1;
	return _valid;
}

std::string BaseFile::fileName() const
{
	return m_fileName;
}

char BaseFile::isValid() const
{
	return _valid;
}

bool BaseFile::FileExists(const std::string & name)
{
	if(!boost::filesystem::exists(name)) {
		std::cout<<"File "<<name<<" doesn't exist!";
		return false;
	}
	return true;
}

void BaseFile::setLatestError(BaseFile::ErrorMsg err)
{
	m_error = err;
}

BaseFile::ErrorMsg BaseFile::latestError() const
{
	return m_error;
}