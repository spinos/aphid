/*
 *  Entity.h
 *  btree
 *
 *  Created by jian zhang on 4/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <iostream>
namespace sdb {

class Entity
{
public:
	Entity(Entity * parent = NULL);
	virtual ~Entity() {}
	
	Entity * parent() const;
	void setParent(Entity * parent);
	bool shareSameParent(Entity * another) const;
	
	virtual void display() const;
private:
	Entity *m_parent;
};

} // end of namespace sdb