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
#include <math/miscfuncs.h>

using namespace aphid;

NoiseRenderer::NoiseRenderer()
{}

NoiseRenderer::~NoiseRenderer()
{}

void NoiseRenderer::traceRays(BufferBlock& rays)
{
	float rgray = RandomF01();
	const int& ns = rays.numSamples();
	for(int i=0;i<ns;++i) {
		float* col = rays.sampleColor(i);
		col[0] = col[1] = col[2] = rgray;
	}
}
