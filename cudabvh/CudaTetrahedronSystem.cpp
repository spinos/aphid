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
#include "tetrahedronSystem_implement.h"

CudaTetrahedronSystem::CudaTetrahedronSystem() {}

CudaTetrahedronSystem::~CudaTetrahedronSystem() {}

void CudaTetrahedronSystem::setDeviceXPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceX = ptr; m_xLoc = loc; }

void CudaTetrahedronSystem::setDeviceXiPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceXi = ptr; m_xiLoc = loc; }

void CudaTetrahedronSystem::setDeviceVPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceV = ptr; m_vLoc = loc; }

void CudaTetrahedronSystem::setDeviceMassPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceMass = ptr; m_massLoc = loc; }

void CudaTetrahedronSystem::setDeviceTretradhedronIndicesPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceTretradhedronIndices = ptr; m_iLoc = loc; }

void CudaTetrahedronSystem::initOnDevice() 
{
    calculateMass();
	setNumLeafNodes(numTetrahedrons());
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
    bvhCalculateLeafAabbsTetrahedron2((Aabb *)dst, (float3 *)cvs, (float3 *)vsrc, 1.f/60.f, (uint4 *)idx, numTetrahedrons());
}

void * CudaTetrahedronSystem::deviceX()
{ return m_deviceX->bufferOnDeviceAt(m_xLoc); }

void * CudaTetrahedronSystem::deviceXi()
{ return m_deviceXi->bufferOnDeviceAt(m_xiLoc); }

void * CudaTetrahedronSystem::deviceV()
{  return m_deviceV->bufferOnDeviceAt(m_vLoc); }

void * CudaTetrahedronSystem::deviceMass()
{  return m_deviceMass->bufferOnDeviceAt(m_massLoc); }

void * CudaTetrahedronSystem::deviceTretradhedronIndices()
{ return m_deviceTretradhedronIndices->bufferOnDeviceAt(m_iLoc); }

void CudaTetrahedronSystem::integrate(float timeStep)
{ tetrahedronSystemIntegrate((float3 *)deviceX(), (float3 *)deviceV(), timeStep, numPoints()); }

void CudaTetrahedronSystem::sendXToHost()
{ m_deviceX->deviceToHost(hostX(), m_xLoc, numPoints() * 12); }

void CudaTetrahedronSystem::sendVToHost()
{ m_deviceV->deviceToHost(hostV(), m_vLoc, numPoints() * 12); }
