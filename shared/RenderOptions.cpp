/*
 *  RenderOptions.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/13/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "RenderOptions.h"

RenderOptions::RenderOptions() 
{
	m_resX = 640;
	m_resY = 480;
	m_AASample = 4;
	m_maxSubdiv = 3;
}

RenderOptions::~RenderOptions() {}

int RenderOptions::AASample() const
{
	return m_AASample;
}

int RenderOptions::renderImageWidth() const
{
	return m_resX;
}

int RenderOptions::renderImageHeight() const
{
	return m_resY;
}

int RenderOptions::maxSubdiv() const
{
	return m_maxSubdiv;
}

void RenderOptions::setAASample(int x)
{
	m_AASample = x;
}

void RenderOptions::setRenderImageWidth(int x)
{
	m_resX = x;
}

void RenderOptions::setRenderImageHeight(int y)
{
	m_resY = y;
}

void RenderOptions::setMaxSubdiv(int x)
{
	m_maxSubdiv = x;
}