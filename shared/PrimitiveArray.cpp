/*
 *  PrimitiveArray.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "PrimitiveArray.h"

unsigned PrimitiveArray::BlockSize = 512*1024/sizeof(Primitive);

PrimitiveArray::PrimitiveArray() : m_pos(0) {}

PrimitiveArray::~PrimitiveArray() 
{
	clear();
}

void PrimitiveArray::clear()
{
	for (std::vector<Primitive *>::iterator it = m_blocks.begin(); 
				it != m_blocks.end(); ++it)
			delete[] *it;
	m_blocks.clear();
	m_pos = 0;
}

void PrimitiveArray::push_back(const Primitive &value) 
{
	unsigned blockIdx = m_pos / BlockSize;
	unsigned offset = m_pos % BlockSize;
	if (blockIdx == m_blocks.size())
		m_blocks.push_back(new Primitive[BlockSize]);
	m_blocks[blockIdx][offset] = value;
	m_pos++;
}

Primitive * PrimitiveArray::allocate(unsigned size) 
{
	unsigned blockIdx = m_pos / BlockSize;
	unsigned offset = m_pos % BlockSize;
	Primitive *result;
	if (offset + size <= BlockSize) {
		if (blockIdx == m_blocks.size())
			m_blocks.push_back(new Primitive[BlockSize]);
		result = m_blocks[blockIdx] + offset;
		m_pos += size;
	} 
	else {
		++blockIdx;
		if (blockIdx == m_blocks.size())
			m_blocks.push_back(new Primitive[BlockSize]);
		result = m_blocks[blockIdx];
		m_pos += BlockSize - offset + size;
	}
	return result;
}

Primitive &PrimitiveArray::operator[](unsigned index) 
{
	return *(m_blocks[index / BlockSize] +
			(index % BlockSize));
}

const Primitive &PrimitiveArray::operator[](unsigned index) const 
{
	return *(m_blocks[index / BlockSize] +
		(index % BlockSize));
}

Triangle *PrimitiveArray::asTriangle(unsigned index)
{
	Primitive prim = *(m_blocks[index / BlockSize] + (index % BlockSize));
	return (Triangle *) prim.geom();
}

Triangle *PrimitiveArray::asTriangle(unsigned index) const
{
	Primitive prim = *(m_blocks[index / BlockSize] + (index % BlockSize));
	return (Triangle *) prim.geom();
}

unsigned PrimitiveArray::size() const 
{
	return m_pos;
}

unsigned PrimitiveArray::blockCount() const 
{
	return m_blocks.size();
}

unsigned PrimitiveArray::capacity() const 
{
	return m_blocks.size() * BlockSize;
}
