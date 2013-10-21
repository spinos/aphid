/*
 *  NamedEntity.cpp
 *  eulerRot
 *
 *  Created by jian zhang on 10/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "NamedEntity.h"

NamedEntity::NamedEntity() { m_name = "unknown"; }
NamedEntity::~NamedEntity() {}
	
void NamedEntity::setName(const std::string & name) { m_name = name; }
std::string NamedEntity::name() const { return m_name; }

void NamedEntity::setIndex(unsigned idx)
{
    m_index = idx;
}

unsigned NamedEntity::index() const
{
    return m_index;
}