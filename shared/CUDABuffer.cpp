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
#include "CudaBase.h"
CUDABuffer::CUDABuffer() : _device_vbo_buffer(0), m_realSize(0) {} 
CUDABuffer::~CUDABuffer() { destroy(); }

void CUDABuffer::create(float * data, unsigned size)
{
	BaseBuffer::create(data, size);
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&_cuda_vbo_resource, bufferName(), cudaGraphicsMapFlagsWriteDiscard)); 
}

void CUDABuffer::create(unsigned size)
{
	if(canResize(size)) return;
	if(size <= m_realSize) {
        setBufferSize(size);
        return;
	}
	destroy();
	m_realSize = minimalMemSize(size);
	CudaBase::MemoryUsed += m_realSize;
	cutilSafeCall(cudaMalloc((void **)&_device_vbo_buffer, m_realSize));
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
		CudaBase::MemoryUsed -= m_realSize;
		m_realSize = 0;
		setBufferSize(0);
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
    cutilSafeCall( cudaMemcpy(_device_vbo_buffer, src, size, cudaMemcpyHostToDevice) );   
}

void CUDABuffer::deviceToHost(void * dst, unsigned size)
{
	cudaError_t err = cudaMemcpy(dst, _device_vbo_buffer, size, cudaMemcpyDeviceToHost); 
	if(err) {
	    std::cout<<"Cudamemcpy threw error\n";
	}
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

const unsigned CUDABuffer::minimalMemSize(unsigned size) const
{ 
// round to 1K
// no less than 4K
    unsigned rds = size;
    rds = ((rds & 1023) == 0) ? rds : ((rds>>10) + 1)<<10;
	return (rds < 4096) ? 4096 : rds;
}

