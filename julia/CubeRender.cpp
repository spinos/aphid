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

using namespace aphid;

CubeRender::CubeRender() {}
CubeRender::~CubeRender() {}

void CubeRender::setSize(const int & w, const int & h)
{
	aphid::CudaRender::setSize(w, h);
	imagebase::resetImage((uint *) colorBuffer(),
                (float *) depthBuffer(),
                512,
                numPixels() );
	CudaBase::CheckCudaError(" reset image");
}

void CubeRender::render()
{
	cuber::render((uint *) colorBuffer(),
                (float *) depthBuffer(),
				16,
				tileX(), tileY() );
	CudaBase::CheckCudaError(" render image");
	colorToHost();
}