/*
 *  NoiseRenderer.h
 * 
 *  noise color on samples 
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef NOISE_RENDERER_H
#define NOISE_RENDERER_H

#include "Renderer.h"

class NoiseRenderer : public Renderer {

public:
	NoiseRenderer();
	virtual ~NoiseRenderer();
	
	virtual void traceRays(BufferBlock& rays);
	
};

#endif
