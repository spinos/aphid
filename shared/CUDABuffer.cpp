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
#include <gl_heads.h>
#include <cutil_gl_inline.h>

CUDABuffer::CUDABuffer() : _device_vbo_buffer(0) {} 
CUDABuffer::~CUDABuffer() {}

void CUDABuffer::create(float * data, unsigned size)
{
	BaseBuffer::create(data, size);
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&_cuda_vbo_resource, bufferName(), cudaGraphicsMapFlagsWriteDiscard)); 
}

void CUDABuffer::create(unsigned size)
{
	cutilSafeCall(cudaMalloc((void **)&_device_vbo_buffer, size));
	setBufferType(kSimple);
}

void CUDABuffer::destroy()
{
    if(bufferType() == kVBO) {
		cudaGraphicsUnregisterResource(_cuda_vbo_resource);
		BaseBuffer::destroyVBO();
	}
	else if(bufferType() == kSimple) {
		if(_device_vbo_buffer == 0) return;
		cudaFree(_device_vbo_buffer);
		_device_vbo_buffer = 0;
	}
}

struct cudaGraphicsResource ** CUDABuffer::resource()
{
    return &_cuda_vbo_resource;
}

void * CUDABuffer::bufferOnDevice() { return _device_vbo_buffer; }

void CUDABuffer::hostToDevice(void * src, unsigned size)
{
	cutilSafeCall( cudaMemcpy(_device_vbo_buffer, src, size, cudaMemcpyHostToDevice) );   
}

void CUDABuffer::deviceToHost(void * dst, unsigned size)
{
	cutilSafeCall( cudaMemcpy(dst, _device_vbo_buffer, size, cudaMemcpyDeviceToHost) );   
}

void CUDABuffer::map(void ** p)
{
	cutilSafeCall(cudaGraphicsMapResources(1, resource(), 0));
	size_t num_bytes; 
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer(p, &num_bytes,  
						       *resource()));
}

void CUDABuffer::unmap()
{
	cutilSafeCall(cudaGraphicsUnmapResources(1, resource(), 0));
}

void CUDABuffer::setDevice()
{
	cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
}