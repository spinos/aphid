/*
 *  ClassificationStorage.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "ClassificationStorage.h"


ClassificationStorage::ClassificationStorage() 
		: m_buffer(0), m_bufferSize(0) { }

ClassificationStorage::~ClassificationStorage() 
{
	clear();
}

void ClassificationStorage::clear()
{
	delete[] m_buffer;
	m_buffer = 0;
}

void ClassificationStorage::setPrimitiveCount(unsigned size) 
{
	clear();
	if (size > 0) {
		m_bufferSize = size/4 + ((size % 4) > 0 ? 1 : 0);
		m_buffer = new char[m_bufferSize];
	} else {
		m_buffer = 0;
	}
}

void ClassificationStorage::set(unsigned index, int value) 
{
	char *ptr = m_buffer + (index >> 2);
	char shift = (index & 3) << 1;
	*ptr = (*ptr & ~(3 << shift)) | (value << shift);
}

int ClassificationStorage::get(unsigned index) const 
{
	char *ptr = m_buffer + (index >> 2);
	char shift = (index & 3) << 1;
	return (*ptr >> shift) & 3;
}

unsigned ClassificationStorage::size() const 
{
	return m_bufferSize;
}
