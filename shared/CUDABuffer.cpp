/*
 *  CUDABuffer.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "CUDABuffer.h"
#include <cutil_inline.h>
#include <cutil_gl_inline.h>

CUDABuffer::CUDABuffer() : _device_vbo_buffer(0) {} 
CUDABuffer::~CUDABuffer() {}

void CUDABuffer::create(float * data, unsigned size)
{
    if(_buffereName != 0) destroy();
	createVBO(data, size);
	if(_buffereName == 0) return;
	
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&_cuda_vbo_resource, _buffereName, cudaGraphicsMapFlagsWriteDiscard));
	cutilSafeCall(cudaMalloc((void **)&_device_vbo_buffer, size));
}

void CUDABuffer::destroy()
{
    cudaGraphicsUnregisterResource(_cuda_vbo_resource);
	destroyVBO();
	if(_device_vbo_buffer == 0) return;
	cudaFree(_device_vbo_buffer);
	_device_vbo_buffer = 0;
}

struct cudaGraphicsResource ** CUDABuffer::resource()
{
    return &_cuda_vbo_resource;
}

void CUDABuffer::setDevice()
{
	cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
}