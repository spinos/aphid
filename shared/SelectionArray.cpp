/*
 *  SelectionArray.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "SelectionArray.h"
#include "Primitive.h"
SelectionArray::SelectionArray() {}
SelectionArray::~SelectionArray() {}

void SelectionArray::reset() 
{
	m_prims.clear();
}

void SelectionArray::add(Primitive * prim)
{
	std::vector<Primitive *>::iterator it;
	for(it = m_prims.begin(); it != m_prims.end(); ++it) {
		if((*it) == prim)
			return;
	}
	m_prims.push_back(prim);
}

unsigned SelectionArray::numPrims() const
{
	return (unsigned)m_prims.size();
}

Primitive * SelectionArray::getPrimitive(const unsigned & idx) const
{
	return m_prims[idx];
}
