/*
 *  RenderContext.h
 *
 *  access to sampler, projector
 *
 *  Created by jian zhang on 2/3/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef RENDER_CONTEXT_H
#define RENDER_CONTEXT_H

class PixelSampler;
class PixelProjector; 

class RenderContext {

	PixelSampler* m_sampler;
	PixelProjector* m_projector;
	
public:

	RenderContext();
	
	void createSampler();
	void createProjector();
	
	const PixelSampler* sampler() const;
	PixelProjector* projector();
	
protected:

private:
	
};

#endif