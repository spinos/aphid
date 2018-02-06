/*
 *  DeepBuffer.cpp
 *  
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DeepBuffer.h"
#include "BufferBlock.h"
#include "DisplayCamera.h"
#include <math/QuickSort.h>
#include <math/miscfuncs.h>

using namespace aphid;

DeepBuffer::DeepBuffer() :
m_width(0),
m_height(0)
{}

DeepBuffer::~DeepBuffer()
{
	std::vector<BufferBlock *>::iterator it = m_blocks.begin();
	for(;it!=m_blocks.end();++it) {
		delete *it;
	}
	m_blocks.clear();
}

void DeepBuffer::create(int w, int h)
{
	m_width = BufferBlock::RoundToBlockSize(w);
	m_height = BufferBlock::RoundToBlockSize(h);
	int nblk = numBlockX() * numBlockY();
	if(nblk > m_blocks.size() ) {
		addBlocks(nblk);
	}
	initBlocks();
	m_priority.reset(new aphid::QuickSortPair<float, int>[nblk] );
}

const int& DeepBuffer::width() const
{ return m_width; }

const int& DeepBuffer::height() const
{ return m_height; }

int DeepBuffer::bufferSize() const
{ return m_width * m_height; }

int DeepBuffer::numBlockX() const
{ return m_width>>BUFFER_BLOCK_TILE_SIZE_P2; }
	
int DeepBuffer::numBlockY() const
{ return m_height>>BUFFER_BLOCK_TILE_SIZE_P2; }

int DeepBuffer::numBlocks() const
{ return numBlockX() * numBlockY(); }

void DeepBuffer::addBlocks(int nblk)
{
	while(m_blocks.size() < nblk) {
		BufferBlock* b = new BufferBlock;
		m_blocks.push_back(b);
	}
}

void DeepBuffer::initBlocks()
{
	const int nblkX = numBlockX();
	const int nblk = numBlocks();
	for(int i=0;i<nblk;++i) {
		int tileY = i / nblkX;
		int tileX = i - tileY * nblkX;
		BufferBlock* bi = block(i);
		bi->setTile(tileX, tileY);
	}
}

BufferBlock* DeepBuffer::block(int i)
{ return m_blocks[i]; }

void DeepBuffer::setBegin(const DisplayCamera* camera)
{
	const int nblk = numBlocks();
	for(int i=0;i<nblk;++i) {
		camera->setBlockView(m_blocks[i]);
		m_blocks[i]->begin();
		
	}
}

BufferBlock* DeepBuffer::highResidualBlock()
{
	const int nblk = numBlocks();
	for(int i=0;i<nblk;++i) {
		QuickSortPair<float, int>& ind = m_priority[i];
		ind.key = m_blocks[i]->residual();
		ind.value = i;
	}
	QuickSort1::Sort<float, int>(m_priority.get(), 0, nblk-1);
	
	return m_blocks[m_priority[nblk - 1].value];
}

int DeepBuffer::findPriorityBegin(const float& thre) const
{
	int low = 0;
	if(m_priority[low].key > thre)
		return low;
	
	int high = numBlocks()-1;
	if(m_priority[high].key < thre)
		return high>>1;
		
	while(high > low+1) {
		int j = (low + high) / 2;
		if(m_priority[j].key < thre) {
			low = j;
		} else {
			high = j;
		}
	}

	return high;
}

float DeepBuffer::maxResidual() const
{
	float r = -1.f;
	const int nblk = numBlocks();
	for(int i=0;i<nblk;++i) {
		const float& ri = m_blocks[i]->residual();
		if(r < ri)
			r = ri;
	}
	return r;
}
