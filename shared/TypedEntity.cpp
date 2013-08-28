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

void TypedEntity::setEntityType(TypeEntries val)
{
    m_type = val;
}

unsigned TypedEntity::entityType() const
{
    return m_type;
}

bool TypedEntity::isTriangleMesh() const
{
    return m_type == TTriangleMesh;
}

bool TypedEntity::isPatchMesh() const
{
    return m_type == TPatchMesh;
}
//:~