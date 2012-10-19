/*
 *  IndexArray.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "IndexArray.h"

unsigned IndexArray::BlockSize = 512*1024/sizeof(unsigned);

IndexArray::IndexArray() : m_pos(0) {}

IndexArray::~IndexArray() 
{
	clear();
}

void IndexArray::clear()
{
	for (std::vector<unsigned *>::iterator it = m_blocks.begin(); 
				it != m_blocks.end(); ++it)
			delete[] *it;
	m_blocks.clear();
	m_pos = 0;
}

void IndexArray::push_back(const unsigned &value) 
{
	unsigned blockIdx = m_pos / BlockSize;
	unsigned offset = m_pos % BlockSize;
	if (blockIdx == m_blocks.size())
		m_blocks.push_back(new unsigned[BlockSize]);
	m_blocks[blockIdx][offset] = value;
	m_pos++;
}

unsigned * IndexArray::allocate(unsigned size) 
{
	unsigned blockIdx = m_pos / BlockSize;
	unsigned offset = m_pos % BlockSize;
	unsigned *result;
	if (offset + size <= BlockSize) {
		if (blockIdx == m_blocks.size())
			m_blocks.push_back(new unsigned[BlockSize]);
		result = m_blocks[blockIdx] + offset;
		m_pos += size;
	} 
	else {
		++blockIdx;
		if (blockIdx == m_blocks.size())
			m_blocks.push_back(new unsigned[BlockSize]);
		result = m_blocks[blockIdx];
		m_pos += BlockSize - offset + size;
	}
	return result;
}

void IndexArray::start()
{
	m_current = 0;
}

void IndexArray::take(const unsigned &value)
{
	*(m_blocks[m_current / BlockSize] +
			(m_current % BlockSize)) = value;
	m_current++;
}

unsigned &IndexArray::operator[](unsigned index) 
{
	return *(m_blocks[index / BlockSize] +
			(index % BlockSize));
}

const unsigned &IndexArray::operator[](unsigned index) const 
{
	return *(m_blocks[index / BlockSize] +
		(index % BlockSize));
}

unsigned IndexArray::size() const 
{
	return m_pos;
}

unsigned IndexArray::blockCount() const 
{
	return m_blocks.size();
}

unsigned IndexArray::capacity() const 
{
	return m_blocks.size() * BlockSize;
}

unsigned IndexArray::taken() const
{
	return m_current;
}
