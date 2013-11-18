/*
 *  BaseImage.cpp
 *  arum
 *
 *  Created by jian zhang on 9/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseImage.h"
#include <iostream>
BaseImage::BaseImage()
{
}

BaseImage::~BaseImage()
{
	clear();
}

void BaseImage::doClear() 
{
	BaseFile::doClear();
}

const char * BaseImage::formatName() const
{
	return "Unknown";
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

void BaseImage::sample(float u, float v, int count, float * dst) const
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

void BaseImage::verbose() const
{
	std::cout<<"Image file: "<<fileName()<<"\n";
	std::cout<<" format: "<<formatName()<<"\n";
	std::cout<<" size: ("<<m_imageWidth<<", "<<m_imageHeight<<")\n";
	if(m_channelRank == RGB)
		std::cout<<" channels: RGB\n";
	else
		std::cout<<" channels: RGBA\n";
	
	if(isOpened())
		std::cout<<" image is verified\n";
}
