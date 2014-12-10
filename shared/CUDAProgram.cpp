/*
 *  CUDAProgram.cpp
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "CUDAProgram.h"
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <iostream>

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
    // cutilSafeCall(cudaGraphicsResourceSetMapFlags(*buffer->resource(), cudaGraphicsMapFlagsNone));
	cutilSafeCall(cudaGraphicsMapResources(1, buffer->resource(), 0));
	size_t num_bytes; 
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer(p, &num_bytes,  
						       *buffer->resource()));
}

void CUDAProgram::unmap(CUDABuffer * buffer)
{
	cutilSafeCall(cudaGraphicsUnmapResources(1, buffer->resource(), 0));
}

void CUDAProgram::calculateDim(unsigned count, unsigned & w, unsigned & h)
{
    w = 4; h = count / w;
    while(w < h)
    {
        w *= 2;
        h = count / w;
    }
}