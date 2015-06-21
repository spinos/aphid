/*
 *  CUDABuffer.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "CUDABuffer.h"
#include <iostream>
#include "CudaBase.h"

CUDABuffer::CUDABuffer() : _device_vbo_buffer(0), m_reseveSize(0),
m_bufferSize(0)
{} 

CUDABuffer::~CUDABuffer() { destroy(); }

void CUDABuffer::create(unsigned size)
{
	if(size <= m_reseveSize) {
        m_bufferSize = size;
        return;
	}
	
	destroy();
	
	m_reseveSize = minimalMemSize(size);
	CudaBase::MemoryUsed += m_reseveSize;
	cudaError_t err = cudaMalloc((void **)&_device_vbo_buffer, m_reseveSize);
	
	CudaBase::CheckCudaError(err, " cu buffer create");
	
	m_bufferSize = size;;
}

void CUDABuffer::destroy()
{
    if(_device_vbo_buffer == 0) return;
	
	cudaError_t err = cudaFree(_device_vbo_buffer);
	CudaBase::CheckCudaError(err, " cu buffer destroy");
	
	CudaBase::MemoryUsed -= m_reseveSize;
	m_reseveSize = 0;
	m_bufferSize = 0;
	_device_vbo_buffer = 0;
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
    cudaError_t err = cudaMemcpy(_device_vbo_buffer, src, size, cudaMemcpyHostToDevice);
    CudaBase::CheckCudaError(err, " cu buffer host to device with size");
}

void CUDABuffer::deviceToHost(void * dst, unsigned size)
{
	//cudaDeviceSynchronize();
	cudaError_t err = cudaMemcpy(dst, _device_vbo_buffer, size, cudaMemcpyDeviceToHost); 
	CudaBase::CheckCudaError(err, " cu buffer device to host with size");
}

void CUDABuffer::hostToDevice(void * src)
{ 
    cudaError_t err = cudaMemcpy(_device_vbo_buffer, src, m_bufferSize, cudaMemcpyHostToDevice); 
    CudaBase::CheckCudaError(err, " cu buffer host to device");
}

void CUDABuffer::deviceToHost(void * dst)
{ 
    //cudaDeviceSynchronize();
	cudaError_t err = cudaMemcpy(dst, _device_vbo_buffer, m_bufferSize, cudaMemcpyDeviceToHost); 
    CudaBase::CheckCudaError(err, " cu buffer device to host");
}

void CUDABuffer::hostToDevice(void * src, unsigned loc, unsigned size)
{
	char * p = (char *)bufferOnDevice();
    p += loc;
	cudaError_t err = cudaMemcpy(p, src, size, cudaMemcpyHostToDevice);
	CudaBase::CheckCudaError(err, " cu buffer host to device with location and size");
}

void CUDABuffer::deviceToHost(void * dst, unsigned loc, unsigned size)
{
    //cudaDeviceSynchronize();
    char * p = (char *)bufferOnDevice();
    p += loc;
    cudaError_t err = cudaMemcpy(dst, p, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout<<" cuda last check point "<<CudaBase::BreakInfo
		<<" error occured when cu buffer device to host with location "
		<<loc<<" and size "<<size;
		CudaBase::CheckCudaError(err, " cu buffer device to host with location and size");
	}
}

const unsigned CUDABuffer::bufferSize() const
{ return m_bufferSize; }

const unsigned CUDABuffer::minimalMemSize(unsigned size) const
{ 
// round to 1K
// no less than 4K
    unsigned rds = size;
    rds = ((rds & 1023) == 0) ? rds : ((rds/1024) + 1) * 1024;
	return (rds < 4096) ? 4096 : rds;
}

