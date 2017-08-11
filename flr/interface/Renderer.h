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

class BufferBlock;

class Renderer {

public:
	Renderer();
	virtual ~Renderer();
	
	virtual void traceRays(BufferBlock& rays);

};

#endif
