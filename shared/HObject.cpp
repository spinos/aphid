/*
 *  HObject.cpp
 *  helloHdf
 *
 *  Created by jian zhang on 6/12/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string.hpp>
#include "HObject.h"
#include "hdf5.h"
#include <sstream>

HDocument HObject::FileIO;

HObject::HObject(const std::string & path)
{
	fObjectPath = ValidPathName(path);
}

char HObject::validate()
{
	if(!exists())
		return 0;
	return 1;
}

char HObject::create()
{
	return 1;
}

char HObject::open()
{
	return 1;
}

std::string HObject::ValidPathName(const std::string & name)
{
	std::string r = name;
	boost::algorithm::replace_all(r, "|", "/");
	boost::algorithm::replace_all(r, "//", "/");
    boost::trim(r);
	return r;
}

std::string HObject::pathToObject() const
{
	return fObjectPath;
}

int HObject::objectType() const
{
	return 0;
}

char HObject::exists()
{
	if(!FileIO.checkExist(fObjectPath))
		return 0;
		
	H5G_stat_t statbuf;

    H5Gget_objinfo(FileIO.fFileId, fObjectPath.c_str(), 0, &statbuf);
	
	if(statbuf.type != objectType())
		return 0;
	return 1;
}

std::string HObject::FullPath(const std::string & entryName, const std::string & sliceName)
{
	std::stringstream sst;
	sst.str("");
	sst<<entryName<<"/"<<sliceName;
	return sst.str();
}

std::string HObject::PartialPath(const std::string & entryName, const std::string & sliceName)
{
	std::string res = sliceName;
	if(res.find(entryName, 0) != 0) return res;
	res.erase(0, entryName.size() + 1);
	return res;
}

std::string HObject::lastName() const
{
	std::string result = fObjectPath;
	int found = result.rfind('/')+1;
	if(found > -1)
		result = result.erase(0, found);

	return result;
}
//:~