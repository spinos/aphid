/*
 *  RenderContext.cpp
 *  
 *
 *  Created by jian zhang on 2/3/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "RenderContext.h"
#include "PixelSampler.h"
#include "PixelProjector.h"

RenderContext::RenderContext()
{}

void RenderContext::createSampler()
{ m_sampler = new PixelSampler; }

void RenderContext::createProjector()
{ m_projector = new PixelProjector; }
	
const PixelSampler* RenderContext::sampler() const
{ return m_sampler; }

PixelProjector* RenderContext::projector()
{ return m_projector; }

