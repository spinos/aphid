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
namespace aphid {

BaseImage::BaseImage() : 
m_isValid(false),
m_fileName("")
{}

BaseImage::BaseImage(const std::string & filename)
{ read(filename); }

BaseImage::~BaseImage()
{}

bool BaseImage::read(const std::string & filename)
{ 
	m_isValid = readImage(filename); 
	m_fileName = "";
	if(m_isValid) m_fileName = filename;
	
	return m_isValid;
}

const bool & BaseImage::isValid() const
{ return m_isValid; }

const std::string & BaseImage::fileName() const
{ return m_fileName; }

bool BaseImage::readImage(const std::string & filename)
{ return false; }

BaseImage::IFormat BaseImage::formatName() const
{ return FUnknown; }

int BaseImage::getWidth() const
{
	return m_imageWidth;
}
	
int BaseImage::getHeight() const
{
	return m_imageHeight;
}

void BaseImage::setWidth(int x)
{ m_imageWidth = x; }

void BaseImage::setHeight(int x)
{ m_imageHeight = x; }

const float BaseImage::aspectRation() const
{ return (float)m_imageHeight/(float)m_imageWidth; }

int BaseImage::pixelLoc(float s, float t, bool flipV, int pixelRank) const
{
	if(flipV) t = 1.f - t;
	int x = m_imageWidth * s;
	int y = m_imageHeight * t;
	if(x < 0) x = 0;
	if(y < 0) y = 0;
	if(x > m_imageWidth - 1) x = m_imageWidth - 1;
	if(y > m_imageHeight - 1) y = m_imageHeight - 1;
	return (y * m_imageWidth + x) * pixelRank;
}

void BaseImage::allWhite()
{}

void BaseImage::allBlack()
{}

void BaseImage::sample(float u, float v, int count, float * dst) const
{}

float BaseImage::sampleRed(float u, float v)
{ return 0.f; }

float BaseImage::sampleRed(float u, float v) const
{ return 0.f; }

void BaseImage::setRed(float u, float v, float red) {}

void BaseImage::applyMask(BaseImage * another) {}

bool BaseImage::isRGBAZ() const 
{ return m_channelRank == RGBAZ; }

void BaseImage::setChannelRank(ChannelRank x)
{ m_channelRank = x; }
	
BaseImage::ChannelRank BaseImage::channelRank() const
{ return m_channelRank; }

std::string BaseImage::formatNameStr() const
{
	if(formatName() < 1) return "unknown";
	return "exr";
}

std::string BaseImage::channelRankStr() const
{
	if(m_channelRank == RGB)
		return "RGB";
	if(m_channelRank == RGBA)
		return "RGBA";
	return "RGBAZ";
}

void BaseImage::verbose() const
{
	if(!isValid() ) {
		std::cout<<"\n invalid image file ";
		return;
	}
		
	std::cout<<"\n image file "<<fileName()
			<<"\n format: "<<formatNameStr()
			<<"\n size: ("<<getWidth()<<", "<<getHeight()
			<<"\n channel: "<<channelRankStr();
	
}

}
