/*
 *  HRFile.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "HRFile.h"
#include <HObject.h>
HRFile::HRFile() : HFile() {}
HRFile::HRFile(const char * name) : HFile(name) {}

bool HRFile::doRead(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadOnly)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	setDocument(HObject::FileIO);

	return true;
}