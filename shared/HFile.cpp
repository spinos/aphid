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
HFile::~HFile() {}

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
    closeOpenedGroups();
	std::cout<<"close "<<HObject::FileIO.fileName()<<"\n";
	HObject::FileIO.close();
	BaseFile::doClose();
}

void HFile::useDocument() 
{
	HObject::FileIO = m_doc;
}

void HFile::setDocument(const HDocument & doc)
{
	m_doc = doc;
}

void HFile::beginWrite()
{ useDocument(); }

void HFile::flush()
{
	useDocument();
	H5Fflush(HObject::FileIO.fFileId, H5F_SCOPE_LOCAL);
}

bool HFile::entityExists(const std::string & name)
{ return m_doc.checkExist(name) == 1; }

bool HFile::find(const std::string & pathName)
{ return entityExists(pathName); }

bool HFile::isGroupOpened(const std::string & pathName) const
{ return m_openedGroups.find(pathName) != m_openedGroups.end(); }

bool HFile::openGroup(const std::string & pathName)
{
    if(!find(pathName)) return false;
    if(isGroupOpened(pathName)) return true;
    m_openedGroups[pathName] = new HBase(pathName);
    return true;
}

void HFile::closeOpenedGroups()
{
    const int n = m_openedGroups.size();
    int i, j;
    for(j=n-1;j>=0;j--) {
        i=0;
        std::map<std::string, HBase *>::iterator it = m_openedGroups.begin();
        for(;it!=m_openedGroups.end();++it) {
            if(i==j) {
                delete it->second;
                break;
            }
            i++;
        }
    }
    m_openedGroups.clear();
}
//:~
