/*
 *  BoxRenderer.cpp
 *  
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BoxRenderer.h"
#include "BufferBlock.h"
#include "RenderContext.h"
#include "PixelSampler.h"
#include <math/Ray.h>
#include <math/BoundingBox.h>
#include "ImageFragment.h"
#include <math/miscfuncs.h>

using namespace aphid;

BoxRenderer::BoxRenderer()
{}

BoxRenderer::~BoxRenderer()
{}

void BoxRenderer::renderFragment(RenderContext& context, BufferBlock& blk)
{
	const PixelSampler* pxsamp = context.sampler();
	pxsamp->generateViewRays(blk);
	
	BoundingBox bx(-8.f, -8.f, -8.f, 8.f, 8.f, 8.f);
	float tmin, tmax;
	
	float col[4];
	Ray viewRay;
	const int& ns = blk.numSamples();
	for(int i=0;i<ns;++i) {
		
		viewRay.set(blk.viewRay(i) );
		
		if(bx.intersect(viewRay, &tmin, &tmax) ) {
			col[0] = col[1] = col[2] = col[3] = 1.f;
		} else {
			col[0] = col[1] = col[2] = col[3] = 0.f;
		}
		
		setFragmentColor(col, i);
	}
	reproject(context, blk);
}
