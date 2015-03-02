/*
 *  CudaBroadphase.cpp
 *  cudabvh
 *
 *  Created by jian zhang on 2/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include "CudaLinearBvh.h"
#include "CudaBroadphase.h"
#include "broadphase_implement.h"
#include "scan_implement.h"
CudaBroadphase::CudaBroadphase() 
{
	m_numObjects = 0;
	m_pairCounts = new CUDABuffer;
	m_scanCounts = new CUDABuffer;
	m_hostPairCounts = new BaseBuffer;
	m_hostScanCounts = new BaseBuffer;
}

CudaBroadphase::~CudaBroadphase() {}

const unsigned CudaBroadphase::numBoxes() const
{ return m_numBoxes; }

void CudaBroadphase::getOverlappingPairCounts(BaseBuffer * dst)
{ m_pairCounts->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaBroadphase::getScanCounts(BaseBuffer * dst)
{ m_scanCounts->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaBroadphase::addBvh(CudaLinearBvh * bvh)
{
	if(m_numObjects==CUDABROADPHASE_MAX_NUMOBJECTS) return;
	m_objects[m_numObjects] = bvh;
	m_numObjects++;
}

void CudaBroadphase::initOnDevice()
{
	if(m_numObjects < 1) return;
	m_objectStart[0] = 0;
	m_numBoxes = 0;
	for(unsigned i = 0; i<m_numObjects; i++) {
		m_objects[i]->initOnDevice();
		m_numBoxes += m_objects[i]->numLeafNodes();
		if(i<m_numObjects-1) {
			m_objectStart[i+1] = m_numBoxes;
		}
	}
	m_scanBufferLength = iDivUp(m_numBoxes, 1024) * 1024;
	m_pairCounts->create(m_scanBufferLength * 4);
	m_scanCounts->create(m_scanBufferLength * 4);
	m_hostPairCounts->create(m_scanBufferLength * 4);
	m_hostScanCounts->create(m_scanBufferLength * 4);
}

void CudaBroadphase::update()
{
	if(m_numObjects < 1) return;
	unsigned i, j;
	for(i = 0; i<m_numObjects; i++)
		m_objects[i]->update();
		
	resetPairCounts();
	for(j = 0; j<m_numObjects; j++) {
		for(i = 0; i<m_numObjects; i++) {
			countOverlappingPairs(j, i);
		}
	}
	prefixSumPairCounts();
	
	const unsigned n = numOverlappings();
	if(n < 1) return;
	
}

void CudaBroadphase::resetPairCounts()
{
	broadphaseResetPairCounts((uint *)m_pairCounts->bufferOnDevice(), m_scanBufferLength);
}

void CudaBroadphase::countOverlappingPairs(unsigned a, unsigned b)
{
	uint * counts = (uint *)m_pairCounts->bufferOnDevice();
	counts += m_objectStart[a];
	
	CudaLinearBvh * query = m_objects[a];
	CudaLinearBvh * tree = m_objects[b];
	
	void * boxes = (Aabb *)query->leafAabbs();
	const unsigned numBoxes = query->numLeafNodes();
	
	void * rootNodeIndex = tree->rootNodeIndex();
	void * internalNodeChildIndex = tree->internalNodeChildIndices();
	void * internalNodeAabbs = tree->internalNodeAabbs();
	void * leafNodeAabbs = tree->leafAabbs();
	void * mortonCodesAndAabbIndices = tree->leafHash();
	
	broadphaseComputePairCounts(counts, (Aabb *)boxes, numBoxes,
							(int *)rootNodeIndex, 
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices,
							(a == b));							
}

void CudaBroadphase::prefixSumPairCounts()
{
    void * scanInput = m_pairCounts->bufferOnDevice();
    void * scanResult = m_scanCounts->bufferOnDevice();
    scanExclusive((uint *)scanResult, (uint *)scanInput, m_scanBufferLength / 1024, 1024);
}

unsigned CudaBroadphase::numOverlappings()
{
    m_pairCounts->deviceToHost(m_hostPairCounts->data(), m_pairCounts->bufferSize());
    m_scanCounts->deviceToHost(m_hostScanCounts->data(), m_scanCounts->bufferSize());
    unsigned * a = (unsigned *)m_hostPairCounts->data();
    unsigned * b = (unsigned *)m_hostScanCounts->data();
    return a[m_scanBufferLength - 1] + b[m_scanBufferLength - 1];   
}

