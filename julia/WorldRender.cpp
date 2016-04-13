/*
 *  WorldRender.cpp
 *  julia
 *
 *  Created by jian zhang on 3/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "WorldRender.h"
#include <CudaBase.h>
#include <cu/ImageBaseInterface.h>
#include "WorldRenderInterface.h"

namespace aphid {

WorldRender::WorldRender(const std::string & filename) :
CudaRender(8)
{
	bool stat = m_io.openWorld(filename);
	if(!stat) {}
}

WorldRender::~WorldRender() 
{}

void WorldRender::setBufferSize(const int & w, const int & h)
{
	aphid::CudaRender::setBufferSize(w, h);
	imagebase::resetImage((uint *) colorBuffer(),
                (float *) nearDepthBuffer(),
				(float *) farDepthBuffer(),
                512,
                w * h );
	CudaBase::CheckCudaError(" world reset image");
}

void WorldRender::render()
{
	CudaNTree<cvx::Box, KdNode4> * tree = m_io.worldTree();
	
    //updateRayFrameVec();
	wldr::setBoxFaces();
	wldr::setRenderRect((int *)&rect() );
    wldr::setFrustum((float *)rayFrameVec());
	wldr::render((uint *) colorBuffer(),
                (float *) nearDepthBuffer(),
				(float *) farDepthBuffer(),
				tree->deviceBranch(),
				tree->deviceLeaf(),
				tree->deviceRope(),
				(int *)tree->deviceIndirection(),
				tree->devicePrim(),
				tileSize(),
				tileX(), tileY() );
	CudaBase::CheckCudaError(" world render image");
	colorToHost();
}

}