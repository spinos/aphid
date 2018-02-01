/*
 *  Renderer.cpp
 *  
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "Renderer.h"
#include "BufferBlock.h"
#include "RenderContext.h"
#include "ImageFragment.h"
#include "PixelProjector.h"

Renderer::Renderer()
{ 
	m_fragment = new ImageFragment;
}

Renderer::~Renderer()
{ 
	delete m_fragment;
}

void Renderer::renderFragment(RenderContext& context, BufferBlock& blk)
{}

void Renderer::reproject(RenderContext& context, BufferBlock& blk)
{
	PixelProjector* prj = context.projector();
	prj->reproject(blk, *m_fragment);
	blk.progressAge();
}

void Renderer::setFragmentColor(const float* x, const int& i)
{ m_fragment->setColor(x, i); }

ImageFragment* Renderer::fragment()
{ return m_fragment; }

float& Renderer::pixelDepthBuffer(const int& i)
{ return m_depth[i]; }
