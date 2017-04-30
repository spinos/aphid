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
#include <CudaReduction.h>
#include <iostream>
#include "masssystem_impl.h"
CudaMassSystem::CudaMassSystem() 
{
	m_initialMass = new CUDABuffer;
	m_nodeEnergy = new CUDABuffer;
	m_reduce = new CudaReduction;
}

CudaMassSystem::~CudaMassSystem() 
{
	delete m_initialMass;
	delete m_nodeEnergy;
	delete m_reduce;
}

void CudaMassSystem::initOnDevice()
{
    m_initialMass->create(numPoints() * 4);
    m_nodeEnergy->create(numPoints() * 4);
	m_initialMass->hostToDevice(hostMass(), numPoints() * 4);
	m_reduce->initOnDevice();
}

void CudaMassSystem::updateSystem(float dt) {}

void CudaMassSystem::resetSystem() {}

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
{ m_deviceAnchor = ptr; m_anchorLoc = loc; }

void CudaMassSystem::setDeviceImpulsePtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceImpulse = ptr; m_impulseLoc = loc; }

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

void * CudaMassSystem::deviceImpulse()
{ return m_deviceImpulse->bufferOnDeviceAt(m_impulseLoc); }

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

float CudaMassSystem::energy()
{ 
    if(isSleeping()) return 0.f;
// mv^2
    void * dst = m_nodeEnergy->bufferOnDevice();
    void * mass = deviceMass();
    void * vel = deviceV();
    masssystem::computeEnergy((float *)dst,
                                (float *)mass,
                                (float3 *)vel,
                                averageNodeMass(),
                                numPoints());
    float e;
    m_reduce->sum<float>(e, (float *)dst, numPoints());
    return e * .001f;
}

float CudaMassSystem::velocitySize()
{ return energy() / totalMass(); }

float CudaMassSystem::impulseSize()
{
    void * dst = m_nodeEnergy->bufferOnDevice();
    void * vel = deviceV();
    masssystem::computeLength((float *)dst,
                                (float3 *)vel,
                                numPoints());
    float e;
    m_reduce->sum<float>(e, (float *)dst, numPoints());
    return e;
}

void CudaMassSystem::stopMoving()
{
    void * vel = deviceV();
    masssystem::zeroVelocity((float3 *)vel,
                                numPoints());
}
//:~