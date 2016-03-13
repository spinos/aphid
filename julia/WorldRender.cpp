/*
 *  WorldRender.cpp
 *  julia
 *
 *  Created by jian zhang on 3/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "WorldRender.h"
#include <CudaNTree.h>
#include <NTreeIO.h>
#include <CudaBase.h>
#include <cu/ImageBaseInterface.h>
#include "CubeRenderInterface.h"

namespace aphid {

WorldRender::WorldRender(const std::string & filename) 
{
	CudaNTree x;
	NTreeIO hio;
	hio.begin(filename, HDocument::oReadOnly );
	
	std::string gridName;
	if(hio.findGrid(gridName))
		std::cout<<"\n found grid "<<gridName;
		
	hio.end();
}

WorldRender::~WorldRender() {}

void WorldRender::setBufferSize(const int & w, const int & h)
{
	aphid::CudaRender::setBufferSize(w, h);
	imagebase::resetImage((uint *) colorBuffer(),
                (float *) depthBuffer(),
                512,
                w * h );
	CudaBase::CheckCudaError(" reset image");
}

void WorldRender::render()
{
    updateRayFrameVec();
	cuber::setBoxFaces();
	cuber::setRenderRect((int *)&rect() );
    cuber::setFrustum((float *)rayFrameVec());
	cuber::render((uint *) colorBuffer(),
                (float *) depthBuffer(),
				16,
				tileX(), tileY() );
	CudaBase::CheckCudaError(" render image");
	colorToHost();
}

}