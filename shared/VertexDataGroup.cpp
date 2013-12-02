/*
 *  VertexDataGroup.cpp
 *  aphid
 *
 *  Created by jian zhang on 12/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "VertexDataGroup.h"

VertexDataGroup::VertexDataGroup() 
{
}

VertexDataGroup::~VertexDataGroup() 
{
	std::vector<float *>::iterator it;
	for(it = m_entries.begin(); it != m_entries.end(); ++it)
	    delete[] *it;
	m_entries.clear();
	m_names.clear();
}

void VertexDataGroup::createEntry(const std::string & name, unsigned num, unsigned fpe)
{
	float *p = new float[num * fpe];
	m_entries.push_back(p);
	m_names.push_back(name);
}

char VertexDataGroup::hasEntry(const std::string & name) const
{
	return entryIdx(name) > -1;
}

float * VertexDataGroup::entry(const std::string & name)
{
	int i = entryIdx(name);
	if(i < 0) return 0;
	return m_entries[i];
}

int VertexDataGroup::entryIdx(const std::string & name) const
{
	std::vector<std::string>::const_iterator it = m_names.begin();
	for(int i= 0;it != m_names.end(); ++it, ++i) {
		if(*it == name) return i;
	}
	return -1;
}