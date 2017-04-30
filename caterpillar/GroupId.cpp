/*
 *  GroupId.cpp
 *  caterpillar
 *
 *  Created by jian zhang on 5/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "GroupId.h"
#include <sstream>
namespace  caterpillar {
GroupId::GroupId() {}
GroupId::~GroupId() 
{
	resetGroups();
	m_data.clear(); 
}

void GroupId::addGroup(const std::string & name)
{
	m_data[name] = std::deque<int>();
}

const bool GroupId::hasGroup(const std::string & name) const
{
	return m_data.find(name) != m_data.end();
}

std::deque<int> & GroupId::group(const std::string & name)
{
	return m_data[name];
}

const std::deque<int> GroupId::getGroup(const std::string & name) const
{
	std::map<std::string, std::deque<int> >::const_iterator it = m_data.find(name);
	if(it == m_data.end()) return std::deque<int>();
	return (*it).second;
}

void GroupId::resetGroups()
{
	std::map<std::string, std::deque<int> >::iterator it = m_data.begin();
	for(; it != m_data.end(); ++it) (*it).second.clear(); 
}

const std::string GroupId::str() const
{
	std::stringstream sst;
	sst.str("");
	
	std::map<std::string, std::deque<int> >::const_iterator it = m_data.begin();
	for(; it != m_data.end(); ++it) {
		sst<<"\""<<(*it).first<<"\": (";
		const std::deque<int> & l = (*it).second;
		for(int i = 0; i < l.size(); i++) {
			sst<<l[i];
			if(i < l.size() - 1) sst<<",";
		}
		sst<<")\n";
	}
	
	return sst.str();
}

const std::deque<std::string > GroupId::getGroupNames() const
{
    std::deque<std::string > res;
    std::map<std::string, std::deque<int> >::const_iterator it = m_data.begin();
	for(; it != m_data.end(); ++it) res.push_back((*it).first);
    return res;
}

}
