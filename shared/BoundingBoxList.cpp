/*
 *  BoundingBoxList.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BoundingBoxList.h"
BoundingBoxList::BoundingBoxList() {m_raw = 0;}
BoundingBoxList::~BoundingBoxList() 
{
	if(m_raw) delete[] m_raw;
}

void BoundingBoxList::create(const unsigned &num)
{
	m_raw = new char[(num / 256 + 1) * 256 * 32 + 31];
	m_aligned = (BoundingBox *)(((unsigned long)m_raw + 32) & (0xffffffff - 31));
}

BoundingBox *BoundingBoxList::ptr()
{
	return m_aligned;
}
