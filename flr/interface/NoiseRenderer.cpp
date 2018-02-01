/*
 *  NoiseRenderer.cpp
 *  
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "NoiseRenderer.h"
#include "BufferBlock.h"
#include "ImageFragment.h"
#include <math/miscfuncs.h>

using namespace aphid;

NoiseRenderer::NoiseRenderer()
{}

NoiseRenderer::~NoiseRenderer()
{}

void NoiseRenderer::renderFragment(RenderContext& context, BufferBlock& blk)
{
	float col[4];
	col[3] = 1.f;
	float rgray = RandomF01();
	col[0] = col[1] = col[2] = rgray;
/// fill renderd fragment	
	const int& ns = fragment()->fragmentWidth() * fragment()->fragmentHeight();
	for(int i=0;i<ns;++i) {
		setFragmentColor(col, i);
	}
	reproject(context, blk);
}
