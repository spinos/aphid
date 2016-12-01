/*
 *  Entity.cpp
 *  btree
 *
 *  Created by jian zhang on 4/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Entity.h"
namespace aphid {

namespace sdb {

Entity::Entity(Entity * parent)
{
	m_parent = parent;
}

Entity * Entity::parent() const
{
	return m_parent;
}

void Entity::setParent(Entity * parent)
{
	m_parent = parent;
}

bool Entity::shareSameParent(Entity * another) const
{
	return parent() == another->parent();
}

void Entity::display() const {}

} // end of namespace sdb

}