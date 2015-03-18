/*
 *  CudaTexture.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *  
 *  http://cuda-programming.blogspot.com/2013/02/cuda-array-in-cuda-how-to-use-cuda.html
 */
#include "CudaTexture.h"
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <iostream>
CudaTexture::CudaTexture() 
{
	m_width = m_height = m_pixelDepth = 0;
	m_texture = 0;
	_cuda_tex_resource = 0;
	m_isHalf = 0;
}
 
CudaTexture::~CudaTexture() {}

void CudaTexture::create(unsigned width, unsigned height, int pixelDepth, bool isHalf)
{
	if(width == m_width && height == m_height && m_pixelDepth == pixelDepth && m_isHalf == isHalf) return;
	destroy();
	
	m_width = width;
	m_height = height;
	m_pixelDepth = pixelDepth;
	m_isHalf = isHalf;
	
	glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB );
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL );
	
    if(isHalf) {
        if(m_pixelDepth == 1)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE16F_ARB, m_width, m_height, 0, GL_RED, GL_HALF_FLOAT_ARB, NULL);
        else if(m_pixelDepth == 3)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, m_width, m_height, 0, GL_RGB, GL_HALF_FLOAT_ARB, NULL);
        else
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, m_width, m_height, 0, GL_RGBA, GL_HALF_FLOAT_ARB, NULL);
    }
    else {
        if(m_pixelDepth == 1)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, m_width, m_height, 0, GL_RED, GL_FLOAT, NULL);
        else if(m_pixelDepth == 3)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, m_width, m_height, 0, GL_RGB, GL_FLOAT, NULL);
        else
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
	}
	
	cutilSafeCall(cudaGraphicsGLRegisterImage(&_cuda_tex_resource, m_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
}

void CudaTexture::destroy()
{
	if(_cuda_tex_resource) cudaGraphicsUnregisterResource(_cuda_tex_resource);
	if(m_texture) glDeleteTextures(1, &m_texture);
	m_texture = 0;
	m_width = m_height = 0;
	_cuda_tex_resource = 0;
}

void CudaTexture::copyFrom(void * src, unsigned size)
{
    std::cout<<" cu tex cpy "<<size<<"\n";
	cudaArray * arr;
	cutilSafeCall(cudaGraphicsMapResources(1, &_cuda_tex_resource, 0));
    cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&arr, _cuda_tex_resource, 0, 0));
	
	cutilSafeCall(cudaMemcpyToArray(arr, 0, 0, src, size, cudaMemcpyDeviceToDevice));

	cudaGraphicsUnmapResources(1, &_cuda_tex_resource, 0);
}

void CudaTexture::bind()
{ glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_texture); }
    
GLuint * CudaTexture::texture()
{ return &m_texture; }

