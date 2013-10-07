/*
 *  BaseFile.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseFile.h"
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>
BaseFile::BaseFile() 
{
	m_fileName = "unknown";
}

BaseFile::BaseFile(const char * name)
{
	m_fileName = std::string(name);
}

std::string BaseFile::fileName() const
{
	return m_fileName;
}

bool BaseFile::FileExists(const std::string & name)
{
	return boost::filesystem::exists(name);
}