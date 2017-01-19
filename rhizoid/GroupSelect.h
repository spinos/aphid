/*
 *  GroupSelect.h
 *  proxyPaint
 *
 *  randomly select an instance among groups
 *
 *  Created by jian zhang on 1/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_GROUP_SELECT_H
#define APH_GROUP_SELECT_H

#include <boost/scoped_array.hpp>
#include <math/ATypes.h>
#include <vector>

namespace aphid {

class GroupSelect {

	boost::scoped_array<int> m_randGroup;
/// (group_count, group_start_index)
	std::vector<Int2> m_groups;
	
public:
	GroupSelect();
	virtual ~GroupSelect();
	
protected:
/// assign a random number to each entity
	void createEntityKeys(int n);
	
	void clearGroups();
	void addGroup(int c);
/// assign start index to each group
	void finishGroups();
	
	const int & entityKey(int i) const;
/// in i-th group select a child by random number k
	int selectInstance(int iGroup, int k) const;
	
private:
};

}
#endif