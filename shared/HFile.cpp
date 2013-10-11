/*
 *  HFile.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "HFile.h"
#include <HObject.h>
#include <iostream>
HFile::HFile() : BaseFile() {}
HFile::HFile(const char * name) : BaseFile(name) {}

bool HFile::create(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oCreate)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	setDocument(HObject::FileIO);

	return BaseFile::create(fileName);
}

bool HFile::open(const std::string & fileName)
{
    if(!FileExists(fileName)) {
		setLatestError(BaseFile::FileNotFound);
		return false;
	}
	
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadAndWrite)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	setDocument(HObject::FileIO);

	return BaseFile::open(fileName);
}

void HFile::useDocument() 
{
	HObject::FileIO = m_doc;
}

void HFile::setDocument(const HDocument & doc)
{
	m_doc = doc;
}

bool HFile::close()
{
	useDocument();
	std::cout<<"close "<<HObject::FileIO.fileName()<<"\n";
	HObject::FileIO.close();
	return BaseFile::close();
}