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
		bi->setEmpty();
	}
}

BufferBlock* DeepBuffer::block(int i)
{ return m_blocks[i]; }
