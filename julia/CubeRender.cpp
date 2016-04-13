/*
 *  CubeRender.cpp
 *  
 *
 *  Created by jian zhang on 3/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include "CubeRender.h"
#include <CudaBase.h>
#include <cu/ImageBaseInterface.h>
#include "CubeRenderInterface.h"

namespace aphid {

CubeRender::CubeRender() 
{ 
	m_test.build(); 
	m_devicePyramidPlanes.reset(new CUDABuffer);
	m_devicePyramidPlanes->create(4000);
	m_devicePyramidBox.reset(new CUDABuffer);
	m_devicePyramidBox->create(4000);
	
	m_devicePyramidPlanes->hostToDevice((char *)m_test.m_pyramid.plane(0), 80);
	m_devicePyramidBox->hostToDevice((char *)m_test.m_pyramid.bbox(), 32);
}

CubeRender::~CubeRender() {}

void CubeRender::setBufferSize(const int & w, const int & h)
{
	CudaRender::setBufferSize(w, h);
	imagebase::resetImage((uint *) colorBuffer(),
                (float *) nearDepthBuffer(),
                512,
                w * h );
	CudaBase::CheckCudaError(" reset image");
}

void CubeRender::render()
{
	cuber::setBoxFaces();
	cuber::setRenderRect((int *)&rect() );
    cuber::setFrustum((float *)rayFrameVec());
	cuber::drawPyramid((uint *) colorBuffer(),
                (float *) nearDepthBuffer(),
				tileSize(),
				tileX(), tileY(),
				m_devicePyramidPlanes->bufferOnDevice(),
				m_devicePyramidBox->bufferOnDevice() );
	CudaBase::CheckCudaError(" render image");
	colorToHost();
}

}