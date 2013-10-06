/*
 *  BaseImage.cpp
 *  arum
 *
 *  Created by jian zhang on 9/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseImage.h"

BaseImage::BaseImage()
{

}

BaseImage::~BaseImage()
{

}

char BaseImage::load(const char *filename)
{
	return 0;
}

void BaseImage::clear()  { _valid = 0; }

char BaseImage::isValid() const
{
	return _valid;
}

int BaseImage::getWidth() const
{
	return m_imageWidth;
}
	
int BaseImage::getHeight() const
{
	return m_imageHeight;
}

int BaseImage::pixelLoc(float s, float t, bool flipV) const
{
	if(flipV) t = 1.f - t;
	int x = m_imageWidth * s;
	int y = m_imageHeight * t;
	if(x < 0) x = 0;
	if(y < 0) y = 0;
	if(x > m_imageWidth - 1) x = m_imageWidth - 1;
	if(y > m_imageHeight - 1) y = m_imageHeight - 1;
	return y * m_imageWidth * m_channelRank + x * m_channelRank;
}

void BaseImage::allWhite()
{

}

void BaseImage::allBlack()
{

}

float BaseImage::sampleRed(float u, float v)
{
	return 0.f;
}

float BaseImage::sampleRed(float u, float v) const
{
	return 0.f;
}

void BaseImage::setRed(float u, float v, float red) {}

void BaseImage::applyMask(BaseImage * another) {}
