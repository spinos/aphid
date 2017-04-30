/*
 *  AssetRender.cpp
 *  
 *
 *  Created by jian zhang on 3/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include "AssetRender.h"
#include <CudaBase.h>
#include <cu/ImageBaseInterface.h>
#include "assetRenderInterface.h"

namespace aphid {

AssetRender::AssetRender() :
CudaRender(8)
{}

AssetRender::~AssetRender() {}

void AssetRender::setBufferSize(const int & w, const int & h)
{
	CudaRender::setBufferSize(w, h);
	imagebase::resetImage((uint *) colorBuffer(),
                (float *) nearDepthBuffer(),
                512,
                w * h );
	CudaBase::CheckCudaError(" reset image");
}

void AssetRender::render()
{
	assr::setRenderRect((int *)&rect() );
    assr::setFrustum((float *)rayFrameVec());

	assr::drawCube((uint *) colorBuffer(),
                (float *) nearDepthBuffer(),
				(float *) farDepthBuffer(),
				tileSize(),
				tileX(), tileY(),
				m_cuTree->deviceBranch(),
				m_cuTree->deviceLeaf(),
				m_cuTree->deviceRope(),
				(int *)m_cuTree->deviceIndirection(),
				m_cuTree->devicePrim() );

	CudaBase::CheckCudaError(" render image");
}

bool AssetRender::load(const std::string & filename, const int & level)
{ 
	if( !m_container.readTree(filename, level) ) return false;
	m_cuTree.reset(new CuTreeT);
	m_cuTree->transfer(m_container.voxelTree() );
	return true;
}

void AssetRender::frameAll()
{
    BoundingBox tb;
    m_container.voxelTree()->getWorldTightBox(&tb);
    const Vector3F coi = tb.center();
    const Vector3F recede = directionToEye() 
                            * (tb.distance(2) + tb.distance(0) );
    setCenterOfInterest(coi);
    setEyePosition(coi + recede);
    updateInvSpace();
    setFarClip(-tb.distance(0)-tb.distance(1)-tb.distance(2));
    updateFrustum();
}

}