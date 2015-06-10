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
#include "TetrahedronSystemInterface.h"
#include <CudaDbgLog.h>

// CudaDbgLog tetsyslg("tetsys.txt");

CudaTetrahedronSystem::CudaTetrahedronSystem()
{
    m_deviceTetrahedronVicinityInd = new CUDABuffer;
	m_deviceTetrahedronVicinityStart = new CUDABuffer;
	m_vicinity = new CUDABuffer;
}

CudaTetrahedronSystem::CudaTetrahedronSystem(ATetrahedronMesh * md) : TetrahedronSystem(md)
{
	m_deviceTetrahedronVicinityInd = new CUDABuffer;
	m_deviceTetrahedronVicinityStart = new CUDABuffer;
	m_vicinity = new CUDABuffer;
}

CudaTetrahedronSystem::~CudaTetrahedronSystem() 
{
	delete m_deviceTetrahedronVicinityInd;
	delete m_deviceTetrahedronVicinityStart;
	delete m_vicinity;
}

void CudaTetrahedronSystem::initOnDevice() 
{
	m_deviceTetrahedronVicinityInd->create(numTetrahedronVicinityInd() * 4);
	m_deviceTetrahedronVicinityStart->create((numTetrahedrons() + 1) * 4);
	m_deviceTetrahedronVicinityInd->hostToDevice(hostTetrahedronVicinityInd());
	m_deviceTetrahedronVicinityStart->hostToDevice(hostTetrahedronVicinityStart());
	m_vicinity->create(numTetrahedrons()*TETRAHEDRONSYSTEM_VICINITY_LENGTH*4);
	
	tetrasys::writeVicinity((int *)vicinity(), 
	         (int *)m_deviceTetrahedronVicinityInd->bufferOnDevice(), 
	         (int *)m_deviceTetrahedronVicinityStart->bufferOnDevice(), 
	         numTetrahedrons());
	
	CudaMassSystem::initOnDevice();
	
	setNumPrimitives(numTetrahedrons());
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
    tetrasys::formTetrahedronAabbs((Aabb *)dst, (float3 *)cvs, (float3 *)vsrc, 1.f/60.f, (uint4 *)idx, numTetrahedrons());
}

void CudaTetrahedronSystem::integrate(float timeStep)
{ tetrahedronSystemIntegrate((float3 *)deviceX(), (float3 *)deviceV(), timeStep, numPoints()); }

void CudaTetrahedronSystem::sendXToHost()
{ deviceXBuf()->deviceToHost(hostX(), xLoc(), numPoints() * 12); }

void CudaTetrahedronSystem::sendVToHost()
{ deviceVBuf()->deviceToHost(hostV(), vLoc(), numPoints() * 12); }

void * CudaTetrahedronSystem::vicinity()
{ return m_vicinity->bufferOnDevice(); }
//:~
