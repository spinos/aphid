/*
 *  Renderer.h
 *  
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef RENDERER_H
#define RENDERER_H

#include "BufferParameters.h"

class BufferBlock;
class RenderContext;
class ImageFragment;

class Renderer {

/// pixel buffer
	ImageFragment* m_fragment;
	float m_depth[MAX_BUFFER_BLOCK_SIZE];
	
public:
	Renderer();
	virtual ~Renderer();

	virtual void renderFragment(RenderContext& context, BufferBlock& blk);

protected:
/// i-th pixel	
	void setFragmentColor(const float* x, const int& i);
	void reproject(RenderContext& context, BufferBlock& blk);
	
	ImageFragment* fragment();
	float& pixelDepthBuffer(const int& i);
	 
};

#endif
