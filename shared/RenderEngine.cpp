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
#include "RenderOptions.h"
RenderEngine::RenderEngine() {}
RenderEngine::~RenderEngine() {}

void RenderEngine::setOptions(RenderOptions * options)
{
	m_options = options;
}

void RenderEngine::setLights(LightGroup * lights)
{
	m_lights = lights;
}

BaseCamera * RenderEngine::camera() const
{
	return m_options->renderCamera();
}

LightGroup * RenderEngine::lights() const
{
	return m_lights;
}

RenderOptions * RenderEngine::options() const
{
	return m_options;
}

void RenderEngine::preRender() {}
void RenderEngine::render() {}
void RenderEngine::postRender() {}

void RenderEngine::startTimer()
{
    m_met.restart();
}

float RenderEngine::elapsedTime()
{
    return m_met.elapsed();
}