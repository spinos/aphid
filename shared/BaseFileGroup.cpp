/*
 *  BaseFileGroup.cpp
 *  
 *
 *  Created by jian zhang on 2/26/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseFileGroup.h"

BaseFileGroup::BaseFileGroup() {}
BaseFileGroup::~BaseFileGroup() 
{ m_files.clear(); }

void BaseFileGroup::addFile(BaseFile * file)
{ m_files[file->fileName()] = file; }

bool BaseFileGroup::getFile(const std::string & name, BaseFile * dst)
{
	if(m_files.find(name) == m_files.end()) return false;
	dst = m_files[name];
	return true;
}
