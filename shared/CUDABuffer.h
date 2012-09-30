/*
 *  CUDABuffer.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BaseBuffer.h>
class CUDABuffer : public BaseBuffer {
public:
	CUDABuffer();
	virtual ~CUDABuffer();
	
	virtual void create(float * data, unsigned size);
	virtual void destroy();
	
	struct cudaGraphicsResource ** resource();
	
	static void setDevice();
	
	struct cudaGraphicsResource *_cuda_vbo_resource;
	void *_device_vbo_buffer;
};

