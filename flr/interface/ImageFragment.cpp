/*
 *  ImageFragment.cpp
 *  
 *
 *  Created by jian zhang on 2/5/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "ImageFragment.h"

ImageFragment::ImageFragment() :
m_fragWidth(BUFFER_BLOCK_TILE_SIZE),
m_fragHeight(BUFFER_BLOCK_TILE_SIZE)
{}

void ImageFragment::setColor(const float* x, const int& i)
{
	m_color[0][i] = x[0];
	m_color[1][i] = x[1];
	m_color[2][i] = x[2];
	m_color[3][i] = x[3];
}

const int& ImageFragment::fragmentWidth() const
{ return m_fragWidth; }

const int& ImageFragment::fragmentHeight() const
{ return m_fragHeight; }

float* ImageFragment::colorComponent(int i)
{ return m_color[i]; }

const float* ImageFragment::colorComponent(int i) const
{ return m_color[i]; }