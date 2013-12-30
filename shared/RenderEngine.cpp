/*
 *  RenderEngine.cpp
 *  aphid
 *
 *  Created by jian zhang on 12/31/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "RenderEngine.h"
#include "BaseCamera.h"
RenderEngine::RenderEngine() {}
RenderEngine::~RenderEngine() {}

void RenderEngine::setCamera(BaseCamera * camera)
{
	m_camera = camera;
}

void RenderEngine::setResolution(unsigned resx, unsigned resy)
{
	m_resolutionX = resx;
	m_resolutionY = resy;
}

unsigned RenderEngine::resolutionX() const
{
	return m_resolutionX;
}
	
unsigned RenderEngine::resolutionY() const
{
	return m_resolutionY;
}

void RenderEngine::render() {}