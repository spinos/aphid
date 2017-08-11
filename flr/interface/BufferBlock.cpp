/*
 *  BufferBlock.cpp
 *  
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BufferBlock.h"
#include "DisplayImage.h"
#include <iostream>

BufferBlock::BufferBlock() :
m_packetX(BUFFER_BLOCK_TILE_SIZE),
m_packetY(BUFFER_BLOCK_TILE_SIZE),
m_numSamples(MAX_BUFFER_BLOCK_SIZE)
{}

BufferBlock::~BufferBlock()
{}

int BufferBlock::BlockSize()
{ return BUFFER_BLOCK_TILE_SIZE; }

int BufferBlock::RoundToBlockSize(int x)
{
	int j = x>>BUFFER_BLOCK_TILE_SIZE_P2;
	if((x & BUFFER_BLOCK_TILE_SIZE_M1) > 0) {
		j++;
	}
	return (j<<BUFFER_BLOCK_TILE_SIZE_P2);
}

void BufferBlock::setEmpty()
{
	memset(m_color, 0, MAX_BUFFER_BLOCK_SIZE_16);
	memset(m_position, 0, MAX_BUFFER_BLOCK_SIZE_12);
}

void BufferBlock::setTile(const int& tx, const int& ty)
{
	m_tileX = tx<<BUFFER_BLOCK_TILE_SIZE_P2;
	m_tileY = ty<<BUFFER_BLOCK_TILE_SIZE_P2;
}

void BufferBlock::projectImage(DisplayImage* img)
{
	for(int j=0;j<m_packetY;++j) {
		uchar* line = img->scanline(j + m_tileY);
		for(int i=0;i<m_packetX;++i) {
			
			float * col = m_color[j * BUFFER_BLOCK_TILE_SIZE + i];
			int r = col[0] * 255; 
			int g = col[1] * 255; 
			int b = col[2] * 255;
			int a = col[4] * 255;
			int icol = (a<<24) | (b<<16) | (g<<8) | r; 
			
			uchar* dst = &line[(i + m_tileX)<<2];
			memcpy(dst, &icol, 4);
			
		}
	}
}

const int& BufferBlock::numSamples() const
{ return m_numSamples; }

float* BufferBlock::sampleColor(int i)
{ return m_color[i]; }

const int& BufferBlock::tileX() const
{ return m_tileX; }

const int& BufferBlock::tileY() const
{ return m_tileY; }

void BufferBlock::setFrame(int i, const float* ori, const float* dir)
{
	memcpy(m_frameOrigin[i], ori, 12);
	memcpy(m_frameDirection[i], dir, 12);
}

void BufferBlock::setNumSamples(int x)
{ m_numSamples = x; }
