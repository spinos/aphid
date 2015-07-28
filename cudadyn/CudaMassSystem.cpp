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
#include <iostream>
CudaMassSystem::CudaMassSystem() 
{
	m_initialMass = new CUDABuffer;
}

CudaMassSystem::~CudaMassSystem() 
{
	delete m_initialMass;
}

void CudaMassSystem::initOnDevice()
{
    m_initialMass->create(numPoints() * 4);
	m_initialMass->hostToDevice(hostMass(), numPoints() * 4);
}

void CudaMassSystem::setDeviceXPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceX = ptr; m_xLoc = loc; }

void CudaMassSystem::setDeviceXiPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceXi = ptr; m_xiLoc = loc; }

void CudaMassSystem::setDeviceVPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceV = ptr; m_vLoc = loc; }

void CudaMassSystem::setDeviceVaPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceVa = ptr; m_vaLoc = loc; }

void CudaMassSystem::setDeviceMassPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceMass = ptr; m_massLoc = loc; }

void CudaMassSystem::setDeviceAnchorPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceAnchor = ptr; m_anchorLoc = loc;}

void CudaMassSystem::setDeviceTretradhedronIndicesPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceTetrahedronIndices = ptr; m_iLoc = loc; }

void * CudaMassSystem::deviceX()
{ return m_deviceX->bufferOnDeviceAt(m_xLoc); }

void * CudaMassSystem::deviceXi()
{ return m_deviceXi->bufferOnDeviceAt(m_xiLoc); }

void * CudaMassSystem::deviceV()
{  return m_deviceV->bufferOnDeviceAt(m_vLoc); }

void * CudaMassSystem::deviceVa()
{ return m_deviceVa->bufferOnDeviceAt(m_vaLoc); }

void * CudaMassSystem::deviceMass()
{ return m_deviceMass->bufferOnDeviceAt(m_massLoc); }

void * CudaMassSystem::deviceInitialMass()
{ return m_initialMass->bufferOnDevice(); }

void * CudaMassSystem::deviceAnchor()
{ return m_deviceAnchor->bufferOnDeviceAt(m_anchorLoc); }

void * CudaMassSystem::deviceTretradhedronIndices()
{ return m_deviceTetrahedronIndices->bufferOnDeviceAt(m_iLoc); }

CUDABuffer * CudaMassSystem::deviceXBuf()
{ return m_deviceX; }

CUDABuffer * CudaMassSystem::deviceVBuf()
{ return m_deviceV; }

CUDABuffer * CudaMassSystem::deviceAnchoredVBuf()
{ return m_deviceVa; }

CUDABuffer * CudaMassSystem::deviceAnchorBuf()
{ return m_deviceAnchor; }

const unsigned CudaMassSystem::xLoc() const
{ return m_xLoc; }

const unsigned CudaMassSystem::vLoc() const
{ return m_vLoc; }

const unsigned CudaMassSystem::anchoredVLoc() const
{ return m_vaLoc; }

CUDABuffer * CudaMassSystem::anchorBuf()
{ return m_deviceAnchor; }

void CudaMassSystem::sendXToHost()
{ deviceXBuf()->deviceToHost(hostX(), xLoc(), numPoints() * 12); }

void CudaMassSystem::sendVToHost()
{ deviceVBuf()->deviceToHost(hostV(), vLoc(), numPoints() * 12); }

void CudaMassSystem::updateMass() {}
//:~