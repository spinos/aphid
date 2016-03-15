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
#include <CudaBase.h>
#include <cu/ImageBaseInterface.h>
#include "CubeRenderInterface.h"

namespace aphid {

WorldRender::WorldRender(const std::string & filename) :
m_worldGrid(NULL)
{
	//CudaNTree x;
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