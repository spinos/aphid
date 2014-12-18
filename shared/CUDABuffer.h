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
	void create(unsigned size);
	
	struct cudaGraphicsResource ** resource();
	void * bufferOnDevice();
	void hostToDevice(void * src, unsigned size);
	void deviceToHost(void * dst, unsigned size);
	
	void map(void ** p);
	void unmap();
	
	static void setDevice();
	
private:
	struct cudaGraphicsResource *_cuda_vbo_resource;
	void *_device_vbo_buffer;
};

