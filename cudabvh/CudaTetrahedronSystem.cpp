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

CudaTetrahedronSystem::CudaTetrahedronSystem() 
{
	m_deviceX = new CUDABuffer;
	m_deviceV = new CUDABuffer;
    m_deviceTretradhedronIndices = new CUDABuffer;
}

CudaTetrahedronSystem::~CudaTetrahedronSystem() 
{
	delete m_deviceX;
	delete m_deviceV;
	delete m_deviceTretradhedronIndices;
}

void CudaTetrahedronSystem::initOnDevice() 
{
	m_deviceX->create(maxNumPoints() * 12);
	m_deviceX->hostToDevice(hostX(), numPoints() * 12);
	m_deviceV->create(maxNumPoints() * 12);
	m_deviceV->hostToDevice(hostV(), numPoints() * 12);
	m_deviceTretradhedronIndices->create(maxNumTetradedrons() * 16);
	m_deviceTretradhedronIndices->hostToDevice(hostTretradhedronIndices(), numTetradedrons() * 16);
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
{  return m_deviceX->bufferOnDevice(); }

void * CudaTetrahedronSystem::deviceV()
{  return m_deviceV->bufferOnDevice(); }

void * CudaTetrahedronSystem::deviceTretradhedronIndices()
{ return m_deviceTretradhedronIndices->bufferOnDevice(); }
