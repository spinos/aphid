/*
 *  CudaTetrahedronSystem.cpp
 *  cudabvh
 *
 *  Created by jian zhang on 2/15/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "CudaTetrahedronSystem.h"
#include <CUDABuffer.h>
#include "createBvh_implement.h"
#include "reduceBox_implement.h"

CudaTetrahedronSystem::CudaTetrahedronSystem() {}

CudaTetrahedronSystem::~CudaTetrahedronSystem() {}

void CudaTetrahedronSystem::setDeviceXPtr(void * ptr)
{ m_deviceX = ptr; }

void CudaTetrahedronSystem::setDeviceVPtr(void * ptr)
{ m_deviceV = ptr; }

void CudaTetrahedronSystem::setDeviceTretradhedronIndicesPtr(void * ptr)
{ m_deviceTretradhedronIndices = ptr; }

void CudaTetrahedronSystem::initOnDevice() 
{
	setNumLeafNodes(numTetradedrons());
	CudaLinearBvh::initOnDevice();
}

void CudaTetrahedronSystem::update()
{
	formTetrahedronAabbs();
    CudaLinearBvh::update();
}

void CudaTetrahedronSystem::formTetrahedronAabbs()
{
	void * cvs = deviceX();
	void * vsrc = deviceV();
    void * idx = deviceTretradhedronIndices();
    void * dst = leafAabbs();
    bvhCalculateLeafAabbsTetrahedron2((Aabb *)dst, (float3 *)cvs, (float3 *)vsrc, 1.f/60.f, (uint4 *)idx, numTetradedrons());
}

void * CudaTetrahedronSystem::deviceX()
{  return m_deviceX; }

void * CudaTetrahedronSystem::deviceV()
{  return m_deviceV; }

void * CudaTetrahedronSystem::deviceTretradhedronIndices()
{ return m_deviceTretradhedronIndices; }
