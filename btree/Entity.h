/*
 *  Entity.h
 *  btree
 *
 *  Created by jian zhang on 4/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma one
#include <iostream>
namespace sdb {
#define MAXPERNODEKEYCOUNT 4
#define MINPERNODEKEYCOUNT 2
class Base
{
public:
};

class Entity : public Base
{
public:
	Entity(Entity * parent = NULL);
	virtual ~Entity() {}
	
	bool isRoot() const;
	bool hasChildren() const;
	bool isLeaf() const;
	bool shareSameParent(Entity * another) const;
	
	Entity * sibling() const;
	Entity * parent() const;
	Entity * firstIndex() const;
	
	void setLeaf();
	void setParent(Entity * parent);
	void connectSibling(Entity * another);
	
	void setFirstIndex(Entity * another);
	
	virtual void display() const;
private:
	Entity *m_first;
	Entity *m_parent;
    bool m_isLeaf;
};
} // end of namespace sdb