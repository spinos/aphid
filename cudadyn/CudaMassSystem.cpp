/*
 *  CudaMassSystem.cpp
 *  testcudafem
 *
 *  Created by jian zhang on 6/9/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "CudaMassSystem.h"
#include <CUDABuffer.h>
CudaMassSystem::CudaMassSystem() 
{
	m_deviceAnchor = new CUDABuffer;
}

CudaMassSystem::~CudaMassSystem() 
{
	delete m_deviceAnchor;
}

void CudaMassSystem::initOnDevice()
{
	m_deviceAnchor->create(numPoints() * 4);
	m_deviceAnchor->hostToDevice(hostAnchor());
}

void CudaMassSystem::setDeviceXPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceX = ptr; m_xLoc = loc; }

void CudaMassSystem::setDeviceXiPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceXi = ptr; m_xiLoc = loc; }

void CudaMassSystem::setDeviceVPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceV = ptr; m_vLoc = loc; }

void CudaMassSystem::setDeviceMassPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceMass = ptr; m_massLoc = loc; }

void CudaMassSystem::setDeviceTretradhedronIndicesPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceTetrahedronIndices = ptr; m_iLoc = loc; }

void * CudaMassSystem::deviceX()
{ return m_deviceX->bufferOnDeviceAt(m_xLoc); }

void * CudaMassSystem::deviceXi()
{ return m_deviceXi->bufferOnDeviceAt(m_xiLoc); }

void * CudaMassSystem::deviceV()
{  return m_deviceV->bufferOnDeviceAt(m_vLoc); }

void * CudaMassSystem::deviceMass()
{  return m_deviceMass->bufferOnDeviceAt(m_massLoc); }

void * CudaMassSystem::deviceTretradhedronIndices()
{ return m_deviceTetrahedronIndices->bufferOnDeviceAt(m_iLoc); }

CUDABuffer * CudaMassSystem::deviceXBuf()
{ return m_deviceX; }
CUDABuffer * CudaMassSystem::deviceVBuf()
{ return m_deviceV; }

const unsigned CudaMassSystem::xLoc() const
{ return m_xLoc; }

const unsigned CudaMassSystem::vLoc() const
{ return m_vLoc; }

void * CudaMassSystem::deviceAnchor()
{ return m_deviceAnchor->bufferOnDevice(); }

CUDABuffer * CudaMassSystem::anchorBuf()
{ return m_deviceAnchor; }
//:~