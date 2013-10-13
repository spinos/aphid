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

bool HFile::doCreate(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oCreate)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	setDocument(HObject::FileIO);
	
	return true;
}

bool HFile::doRead(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadAndWrite)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	setDocument(HObject::FileIO);

	return true;
}

void HFile::doClose()
{
	if(!isOpened()) return;
	useDocument();
	std::cout<<"close "<<HObject::FileIO.fileName()<<"\n";
	HObject::FileIO.close();
}

void HFile::useDocument() 
{
	HObject::FileIO = m_doc;
}

void HFile::setDocument(const HDocument & doc)
{
	m_doc = doc;
}

void HFile::flush()
{
	useDocument();
	H5Fflush(HObject::FileIO.fFileId, H5F_SCOPE_LOCAL);
}
