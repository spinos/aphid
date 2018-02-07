/*
 *  PixelSampler.cpp
 *  
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "PixelSampler.h"
#include "BufferBlock.h"
#include "DisplayCamera.h"
#include <math/miscfuncs.h>
#include <math/haltonSequence.h>

using namespace aphid;

PixelSampler::PixelSampler()
{}

PixelSampler::~PixelSampler()
{}

void PixelSampler::generateViewRays(BufferBlock& blk) const
{
	const int& n = blk.packetX();
	const int& m = blk.packetY();
	float dx = 1.f / (float)n;
	float dy = 1.f / (float)m;
	float u, v;
	for(int j=0;j<m;++j) {
		for(int i=0;i<n;++i) {
			const float* h32 = Halton32Sequence256[rand() & 255];
			u = dx * (h32[0] + i);
			v = dy * (h32[1] + j);
			blk.calculateViewRay(u, v, j * n + i);
		}
	}
}
