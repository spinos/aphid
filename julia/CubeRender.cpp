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
{ m_test.build(); }

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
    updateRayFrameVec();
	cuber::setBoxFaces();
	cuber::setRenderRect((int *)&rect() );
    cuber::setFrustum((float *)rayFrameVec());
	cuber::render((uint *) colorBuffer(),
                (float *) nearDepthBuffer(),
				tileSize(),
				tileX(), tileY() );
	CudaBase::CheckCudaError(" render image");
	colorToHost();
}

}