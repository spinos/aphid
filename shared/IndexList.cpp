/*
 *  IndexList.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "IndexList.h"
#include <vector>

IndexList::IndexList() {m_raw = 0;}
IndexList::~IndexList() 
{
	if(m_raw) delete[] m_raw;
}

void IndexList::create(const unsigned &num)
{
	m_raw = new char[(num / 256 + 1) * 256 * 4 + 31];
    m_aligned = (unsigned *)m_raw;
	//m_aligned = (unsigned *)((reinterpret_cast<uintptr_t>(m_raw) + 32) & (0xffffffff - 31));
}

unsigned *IndexList::ptr()
{
	return m_aligned;
}
