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
CudaRender(8),
m_worldGrid(NULL)
{
	bool stat = m_io.begin(filename, HDocument::oReadOnly );
	if(!stat)
		return;
	
	std::string gridName;
	stat = m_io.findGrid(gridName);
	if(!stat)
		return;
	
	std::cout<<"\n found grid "<<gridName;
	
	m_worldGrid = new WorldGridT(gridName);
	m_worldGrid->load();
	m_io.loadGridCoord<WorldGridT >(&m_worldCoord, m_worldGrid);
	
	std::string treeName;
	stat = m_io.findTree(treeName, gridName);
	if(!stat)
		return;
		
	std::cout<<"\n found tree "<<treeName;

	m_worldTree = new WorldTreeT(treeName);
	m_worldTree->load();
	m_worldTree->close();
	m_worldTree->setSource(&m_worldCoord);
}

WorldRender::~WorldRender() 
{ 
	if(m_worldGrid) m_worldGrid->close();
	m_io.end(); 
}

void WorldRender::setBufferSize(const int & w, const int & h)
{
	aphid::CudaRender::setBufferSize(w, h);
	imagebase::resetImage((uint *) colorBuffer(),
                (float *) nearDepthBuffer(),
				(float *) farDepthBuffer(),
                512,
                w * h );
	CudaBase::CheckCudaError(" reset image");
}

void WorldRender::render()
{
    updateRayFrameVec();
	wldr::setBoxFaces();
	wldr::setRenderRect((int *)&rect() );
    wldr::setFrustum((float *)rayFrameVec());
	wldr::render((uint *) colorBuffer(),
                (float *) nearDepthBuffer(),
				(float *) farDepthBuffer(),
				m_worldTree->deviceBranch(),
				m_worldTree->deviceLeaf(),
				m_worldTree->deviceRope(),
				(int *)m_worldTree->deviceIndirection(),
				m_worldTree->devicePrim(),
				tileSize(),
				tileX(), tileY() );
	CudaBase::CheckCudaError(" render image");
	colorToHost();
}

}