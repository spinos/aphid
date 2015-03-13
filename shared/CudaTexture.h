/*
 *  CUDABuffer.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <gl_heads.h>
#include <cuda_runtime_api.h>
class CudaTexture {
public:
	CudaTexture();
	virtual ~CudaTexture();
	
	void create(unsigned width, unsigned height, int pixelDepth);
	void destroy();
	
	void copyFrom(void * src, unsigned size);
	void map(cudaArray * p);
	void unmap();
	
	void bind();
	
private:
	GLuint m_texture;
	unsigned m_width, m_height, m_pixelDepth;
	cudaGraphicsResource_t _cuda_tex_resource;
};

