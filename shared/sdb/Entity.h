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

namespace aphid {

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

template<typename T>
class Single : public Entity
{
public:
	Single(Entity * parent = NULL) : Entity(parent),
	m_p(NULL)
	{}
	
	T * data() {
		return m_p;
	}
	
	void setData(T * x) {
		m_p = x;
	}
	
private:
	T * m_p;
};

} // end of namespace sdb

}