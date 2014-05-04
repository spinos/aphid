/*
 *  Entity.cpp
 *  btree
 *
 *  Created by jian zhang on 4/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Entity.h"
namespace sdb {

Entity::Entity(Entity * parent)
{
	m_parent = parent;
    m_first = NULL;
	m_isLeaf = false;
}

bool Entity::isRoot() const { return m_parent == NULL; }

bool Entity::hasChildren() const 
{ 
	if(isLeaf()) return false; 
	return m_first != NULL; 
}

bool Entity::isLeaf() const { return m_isLeaf; }




bool Entity::shareSameParent(Entity * another) const
{
	return parent() == another->parent();
}



Entity * Entity::sibling() const
{
	return m_first;
}

Entity * Entity::parent() const
{
	return m_parent;
}

Entity * Entity::firstIndex() const { return m_first; }

void Entity::setLeaf() { m_isLeaf = true; }

void Entity::setParent(Entity * parent)
{
	m_parent = parent;
}

void Entity::connectSibling(Entity * another)
{
	m_first = another;
}

void Entity::setFirstIndex(Entity * another)
{
	m_first = another;
}

void Entity::display() const {}

} // end of namespace sdb