/*
 *  GroupId.h
 *  caterpillar
 *
 *  Created by jian zhang on 5/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <string>
#include <deque>
#include <map>
namespace caterpillar {
class GroupId {
public:
	GroupId();
	virtual ~GroupId();
	
	void addGroup(const std::string & name);
	const bool hasGroup(const std::string & name) const;
	std::deque<int> & group(const std::string & name);
	void resetGroups();
	const std::string str() const;
private:
	std::map<std::string, std::deque<int> > m_data;
};
}