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
#include "LightGroup.h"
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

void RenderEngine::setLights(LightGroup * lights)
{
	m_lights = lights;
}

unsigned RenderEngine::resolutionX() const
{
	return m_resolutionX;
}
	
unsigned RenderEngine::resolutionY() const
{
	return m_resolutionY;
}

BaseCamera * RenderEngine::camera() const
{
	return m_camera;
}

LightGroup * RenderEngine::lights() const
{
	return m_lights;
}

void RenderEngine::preRender() {}
void RenderEngine::render() {}
void RenderEngine::postRender() {}