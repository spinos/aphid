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
#include "ImageFragment.h"
#include <math/miscfuncs.h>
#include <iostream>

using namespace aphid;

BufferBlock::BufferBlock() :
m_packetX(BUFFER_BLOCK_TILE_SIZE),
m_packetY(BUFFER_BLOCK_TILE_SIZE),
m_numSamples(MAX_BUFFER_BLOCK_SIZE)
{
	m_fragment = new ImageFragment;
}

BufferBlock::~BufferBlock()
{
	delete m_fragment;
}

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

void BufferBlock::setTile(const int& tx, const int& ty)
{
	m_tileX = tx<<BUFFER_BLOCK_TILE_SIZE_P2;
	m_tileY = ty<<BUFFER_BLOCK_TILE_SIZE_P2;
}

void BufferBlock::projectImage(DisplayImage* img)
{
	const float* rfrag = m_fragment->colorComponent(0);
	const float* gfrag = m_fragment->colorComponent(1);
	const float* bfrag = m_fragment->colorComponent(2);
	const float* afrag = m_fragment->colorComponent(3);
	
	for(int j=0;j<m_packetY;++j) {
		uchar* line = img->scanline(j + m_tileY);
		for(int i=0;i<m_packetX;++i) {
			
			int ind = j * BUFFER_BLOCK_TILE_SIZE + i;
			int r = rfrag[ind] * 255; 
			int g = gfrag[ind] * 255; 
			int b = bfrag[ind] * 255;
			int a = afrag[ind] * 255;
			int icol = (a<<24) | (b<<16) | (g<<8) | r; 
			
			uchar* dst = &line[(i + m_tileX)<<2];
			memcpy(dst, &icol, 4);
			
		}
	}
}

const int& BufferBlock::numSamples() const
{ return m_numSamples; }

ImageFragment* BufferBlock::fragment()
{ return m_fragment; }

const int& BufferBlock::tileX() const
{ return m_tileX; }

const int& BufferBlock::tileY() const
{ return m_tileY; }

const int& BufferBlock::packetX() const
{ return m_packetX; }

const int& BufferBlock::packetY() const
{ return m_packetY; }

const float& BufferBlock::residual() const
{ return m_residual; }

void BufferBlock::setFrame(int i, const float* ori, const float* dir)
{
	memcpy(m_frame[i], ori, 12);
	memcpy(&m_frame[i][3], dir, 12);
}

void BufferBlock::setNumSamples(int x)
{ m_numSamples = x; }

void BufferBlock::calculateViewRay(const float& u, const float& v,
						const int& i)
{
	float* rayi = m_viewRay[i];
	float mx[2][6];
	for(int d=0;d<6;++d) {
		mx[0][d] = (1.f - u) * m_frame[0][d] + u * m_frame[1][d];
	}
	for(int d=0;d<6;++d) {
		mx[1][d] = (1.f - u) * m_frame[2][d] + u * m_frame[3][d];
	}
	for(int d=0;d<6;++d) {
		rayi[d] = (1.f - v) * mx[0][d] + v * mx[1][d];
	}
	
	float l = 1.f / sqrt(rayi[3] * rayi[3] + rayi[4] * rayi[4] + rayi[5] * rayi[5]);
	rayi[3] *= l;
	rayi[4] *= l;
	rayi[5] *= l;
	rayi[6] = 0.f;
	rayi[7] = 1e10f;
}

const float* BufferBlock::viewRay(const int& i) const
{ return m_viewRay[i]; }

void BufferBlock::begin()
{ 
	m_age = 0;
	m_residual = RandomFlh(1000.f, 199999.f);  
}

void BufferBlock::progressAge()
{ m_age++; }

const int& BufferBlock::age() const
{ return m_age; }

void BufferBlock::setResidual(const float& x)
{ m_residual = x; }
