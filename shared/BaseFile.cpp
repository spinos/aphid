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

bool BaseFile::create(const std::string & filename)
{
	m_fileName = filename;
	_valid = true;
	return _valid;
}

bool BaseFile::open(const std::string & filename)
{
	m_fileName = filename;
	_valid = true;
	return _valid;
}

bool BaseFile::open()
{
    if(isUntitled()) return false;
    return open(m_fileName);
}

bool BaseFile::save()
{
    return true;
}

bool BaseFile::close()
{
    return true;
}

void BaseFile::setFileName(const std::string & filename)
{
    m_fileName = filename;
}

std::string BaseFile::fileName() const
{
	return m_fileName;
}

void BaseFile::setValid()
{
    _valid = true;
}

void BaseFile::setInvalid()
{
    _valid = false;
}

bool BaseFile::isValid() const
{
	return _valid;
}

bool BaseFile::isUntitled() const
{
	return m_fileName == "untitled";
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