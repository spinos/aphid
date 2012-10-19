/*
 *  Primitive.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "Primitive.h"

Primitive::Primitive() {}

void Primitive::setType(short t)
{
	m_type = t;
}

const short &Primitive::getType() const
{
	return m_type;
}