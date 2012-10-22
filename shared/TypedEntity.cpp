/*
 *  TypedEntity.cpp
 *  
 *
 *  Created by jian zhang on 10/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "TypedEntity.h"

TypedEntity::TypedEntity() {}
void TypedEntity::setMeshType()
{
	m_type = 0;
}

bool TypedEntity::isMesh() const
{
	return m_type == 0;
}