/*
 *  CUDABuffer.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <gl_heads.h>
#include "CUDABuffer.h"
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <iostream>
CUDABuffer::CUDABuffer() : _device_vbo_buffer(0) {} 
CUDABuffer::~CUDABuffer() { destroy(); }

void CUDABuffer::create(float * data, unsigned size)
{
	BaseBuffer::create(data, size);
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&_cuda_vbo_resource, bufferName(), cudaGraphicsMapFlagsWriteDiscard)); 
}

void CUDABuffer::create(unsigned size)
{
	if(canResize(size)) return;
	destroy();
	// std::cout<<"cu create buf "<<size<<" \n";
	cutilSafeCall(cudaMalloc((void **)&_device_vbo_buffer, size));
	setBufferType(kOnDevice);
	setBufferSize(size);
}

void CUDABuffer::destroy()
{
    if(bufferType() == kVBO) {
		cudaGraphicsUnregisterResource(_cuda_vbo_resource);
		BaseBuffer::destroyVBO();
	}
	else if(bufferType() == kOnDevice) {
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

void * CUDABuffer::bufferOnDeviceAt(unsigned loc)
{
	char * p = (char *)bufferOnDevice();
    p += loc;
	return p;
}

void CUDABuffer::hostToDevice(void * src, unsigned size)
{
    cudaMemcpy(_device_vbo_buffer, src, size, cudaMemcpyHostToDevice);   
}

void CUDABuffer::deviceToHost(void * dst, unsigned size)
{
	cutilSafeCall( cudaMemcpy(dst, _device_vbo_buffer, size, cudaMemcpyDeviceToHost) );   
}

void CUDABuffer::hostToDevice(void * src)
{ cutilSafeCall( cudaMemcpy(_device_vbo_buffer, src, bufferSize(), cudaMemcpyHostToDevice) ); }

void CUDABuffer::deviceToHost(void * dst)
{ cutilSafeCall( cudaMemcpy(dst, _device_vbo_buffer, bufferSize(), cudaMemcpyDeviceToHost) ); }

void CUDABuffer::hostToDevice(void * src, unsigned loc, unsigned size)
{
	char * p = (char *)bufferOnDevice();
    p += loc;
	cudaMemcpy(p, src, size, cudaMemcpyHostToDevice);
}

void CUDABuffer::deviceToHost(void * dst, unsigned loc, unsigned size)
{
    char * p = (char *)bufferOnDevice();
    p += loc;
    cudaMemcpy(dst, p, size, cudaMemcpyDeviceToHost);
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
