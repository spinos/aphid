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
	std::vector<char *>::iterator it;
	for(it = m_entries.begin(); it != m_entries.end(); ++it)
	    delete[] *it;
	m_entries.clear();
	m_names.clear();
}

void VertexDataGroup::createEntry(const std::string & name, unsigned num, short bpe)
{
	char *p = new char[num * bpe];
	m_entries.push_back(p);
	NameAndType nat;
	nat._name = name;
	nat._type = bpe;
	m_names.push_back(nat);
}

char VertexDataGroup::hasEntry(const std::string & name) const
{
	return entryIdx(name) > -1;
}

char * VertexDataGroup::entry(const std::string & name)
{
	int i = entryIdx(name);
	if(i < 0) return 0;
	return m_entries[i];
}

char * VertexDataGroup::entry(const unsigned & idx, std::string & name, short & bpe)
{
	name = m_names[idx]._name;
	bpe = m_names[idx]._type;
	return m_entries[idx];
}

unsigned VertexDataGroup::fpe(const std::string & name)
{
	int i = entryIdx(name);
	if(i < 0) return 0;
	return m_names[i]._type;
}

int VertexDataGroup::entryIdx(const std::string & name) const
{
	std::vector<NameAndType>::const_iterator it = m_names.begin();
	for(int i= 0;it != m_names.end(); ++it, ++i) {
		if((*it)._name == name) return i;
	}
	return -1;
}

unsigned VertexDataGroup::numEntries() const
{
	return m_names.size();
}

