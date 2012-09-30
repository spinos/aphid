/*
 *  CUDAProgram.cpp
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "CUDAProgram.h"

CUDAProgram::CUDAProgram() {}
CUDAProgram::~CUDAProgram() {}

void CUDAProgram::run(CUDABuffer * buffer)
{	
	// float3 *dptr;
	// map(buffer, (void **)&dptr);
	// unmap(buffer);
}

void CUDAProgram::map(CUDABuffer * buffer, void ** p)
{
	cutilSafeCall(cudaGraphicsMapResources(1, buffer->resource(), 0));
	size_t num_bytes; 
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer(p, &num_bytes,  
						       *buffer->resource()));
}

void CUDAProgram::unmap(CUDABuffer * buffer)
{
	cutilSafeCall(cudaGraphicsUnmapResources(1, buffer->resource(), 0));
}