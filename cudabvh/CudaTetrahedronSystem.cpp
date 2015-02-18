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
    m_deviceTretradhedronIndices = new CUDABuffer;
}

CudaTetrahedronSystem::~CudaTetrahedronSystem() 
{
	delete m_deviceX;
	delete m_deviceTretradhedronIndices;
}

void CudaTetrahedronSystem::initOnDevice() 
{
	m_deviceX->create(maxNumPoints() * 12);
	m_deviceX->hostToDevice(hostX(), numPoints() * 12);
	m_deviceTretradhedronIndices->create(maxNumTetradedrons() * 16);
	m_deviceTretradhedronIndices->hostToDevice(hostTretradhedronIndices(), numTetradedrons() * 16);
	setNumLeafNodes(numTetradedrons());
	CudaLinearBvh::initOnDevice();
}

void CudaTetrahedronSystem::update()
{
	formTetrahedronAabbs();
    combineAabbFirst();
	CudaLinearBvh::update();
}

void CudaTetrahedronSystem::formTetrahedronAabbs()
{
	void * cvs = deviceX();
    void * idx = deviceTretradhedronIndices();
    void * dst = leafAabbs();
    bvhCalculateLeafAabbsTetrahedron((Aabb *)dst, (float3 *)cvs, (uint4 *)idx, numTetradedrons());
}

void CudaTetrahedronSystem::combineAabbFirst()
{
	void * psrc = deviceX();
    void * pdst = combineAabbsBuffer();
	
	unsigned n = nextPow2(numPoints());
	unsigned threads, blocks;
	getReduceBlockThread(blocks, threads, n);
	
	// std::cout<<"n0 "<<n<<" blocks x threads : "<<blocks<<" x "<<threads<<" sharedmem size "<<threads * sizeof(Aabb)<<"\n";
	
	bvhReduceAabbByPoints((Aabb *)pdst, (float3 *)psrc, n, blocks, threads, numPoints());
	
	setCombineAabbSecondBlocks(blocks);
}

void * CudaTetrahedronSystem::deviceX()
{  return m_deviceX->bufferOnDevice(); }

void * CudaTetrahedronSystem::deviceTretradhedronIndices()
{ return m_deviceTretradhedronIndices->bufferOnDevice(); }
