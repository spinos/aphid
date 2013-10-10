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
	return true;
}