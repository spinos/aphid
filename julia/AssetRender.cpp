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
CudaRender(16)
{ 
	m_test.build(); 
	m_devicePyramidPlanes.reset(new CUDABuffer);
	m_devicePyramidPlanes->create(4000);
	m_devicePyramidBox.reset(new CUDABuffer);
	m_devicePyramidBox->create(4000);
	
	m_devicePyramidPlanes->hostToDevice((char *)m_test.m_pyramid.plane(0), 80);
	m_devicePyramidBox->hostToDevice((char *)m_test.m_pyramid.bbox(), 32);
}

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

	assr::drawPyramid((uint *) colorBuffer(),
                (float *) nearDepthBuffer(),
				tileSize(),
				tileX(), tileY(),
				m_devicePyramidPlanes->bufferOnDevice(),
				m_devicePyramidBox->bufferOnDevice() );

	CudaBase::CheckCudaError(" render image");
}

bool AssetRender::load(const std::string & filename, const int & level)
{ 
	if( !m_container.readTree(filename, level) ) return false;
	m_cuTree.reset(new CuTreeT);
	m_cuTree->transfer(m_container.voxelTree() );
	return true;
}

}