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
#include <CudaBase.h>
#include <ScanUtil.h>
#include <CudaTetrahedronSystem.h>
CudaBroadphase::CudaBroadphase() 
{
    m_numObjects = 0;
	m_pairCacheLength = 0;
	m_pairCounts = new CUDABuffer;
	m_pairStart = new CUDABuffer;
	m_scanIntermediate = new CUDABuffer;
	m_pairCache = new CUDABuffer;
}

CudaBroadphase::~CudaBroadphase() {}

const unsigned CudaBroadphase::numBoxes() const
{ return m_numBoxes; }

const unsigned CudaBroadphase::pairCacheLength() const
{ return m_pairCacheLength; }

const unsigned CudaBroadphase::numOverlappingPairs() const
{ return m_pairCacheLength; }

const unsigned CudaBroadphase::objectStart(unsigned ind) const
{ return m_objectStart[ind]; }

void CudaBroadphase::getOverlappingPairCounts(BaseBuffer * dst)
{ m_pairCounts->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaBroadphase::getOverlappingPairCache(BaseBuffer * dst)
{ m_pairCache->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaBroadphase::getScanCounts(BaseBuffer * dst)
{ m_pairStart->deviceToHost(dst->data(), dst->bufferSize()); }

CUDABuffer * CudaBroadphase::overlappingPairBuf()
{ return m_pairCache; }

void CudaBroadphase::addBvh(CudaLinearBvh * bvh)
{
	if(m_numObjects==CUDABROADPHASE_MAX_NUMOBJECTS) return;
	m_objects[m_numObjects] = bvh;
	m_numObjects++;
}

const unsigned CudaBroadphase::numObjects() const
{ return m_numObjects; }

CudaLinearBvh * CudaBroadphase::object(unsigned i) const
{ return m_objects[i]; }

void CudaBroadphase::initOnDevice()
{
	if(m_numObjects < 1) return;
	m_objectStart[0] = 0;
	m_numBoxes = 0;
	for(unsigned i = 0; i<m_numObjects; i++) {
		m_numBoxes += m_objects[i]->numLeafNodes();
		if(i<m_numObjects-1) {
			m_objectStart[i+1] = m_numBoxes;
		}
	}
	
	m_scanBufferLength = iDivUp(m_numBoxes, 1024) * 1024;
	m_pairCounts->create(m_scanBufferLength * 4);
	m_pairStart->create(m_scanBufferLength * 4);
	m_scanIntermediate->create(m_scanBufferLength * 4);
}

void CudaBroadphase::computeOverlappingPairs()
{
	if(m_numObjects < 1) return;
	unsigned i, j;
	
	resetPairCounts();
	for(j = 0; j<m_numObjects; j++) {
		for(i = j; i<m_numObjects; i++) {
			countOverlappingPairs(j, i);
		}
	}
	
	prefixSumPairCounts();
	
	m_pairCacheLength = // getScanResult(m_pairCounts, m_pairStart, m_scanBufferLength - 1);
	ScanUtil::getScanResult(m_pairCounts, m_pairStart, m_scanBufferLength);
	if(m_pairCacheLength < 1) return;
	
	m_pairCache->create(nextPow2(m_pairCacheLength) * 8);
	
	void * cache = m_pairCache->bufferOnDevice();
	broadphaseResetPairCache((uint2 *)cache, nextPow2(m_pairCacheLength));
	
	for(j = 0; j<m_numObjects; j++) {
		for(i = j; i<m_numObjects; i++) {
			writeOverlappingPairs(j, i);
		}
	}
}

void CudaBroadphase::resetPairCounts()
{
	broadphaseResetPairCounts((uint *)m_pairCounts->bufferOnDevice(), m_scanBufferLength);
}

void CudaBroadphase::countOverlappingPairs(unsigned a, unsigned b)
{
    if(a == b) countOverlappingPairsSelf(a);
    else countOverlappingPairsOther(a, b);
}

void CudaBroadphase::countOverlappingPairsSelf(unsigned a)
{
    uint * counts = (uint *)m_pairCounts->bufferOnDevice();
	counts += m_objectStart[a];

// only for tetrahedron system
	CudaTetrahedronSystem * query = static_cast<CudaTetrahedronSystem *>(m_objects[a]);
	CudaLinearBvh * tree = m_objects[a];
	
	void * boxes = (Aabb *)query->leafAabbs();
	const unsigned numBoxes = query->numLeafNodes();
	void * exclusionInd = query->deviceVicinityInd();
	void * exclusionStart = query->deviceVicinityStart();
	
	void * rootNodeIndex = tree->rootNodeIndex();
	void * internalNodeChildIndex = tree->internalNodeChildIndices();
	void * internalNodeAabbs = tree->internalNodeAabbs();
	void * leafNodeAabbs = tree->leafAabbs();
	void * mortonCodesAndAabbIndices = tree->leafHash();
	
	broadphaseComputePairCountsSelfCollideExclusion(counts, (Aabb *)boxes, numBoxes,
							(int *)rootNodeIndex, 
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices,
							(uint *)exclusionInd,
							(uint *)exclusionStart);
}

void CudaBroadphase::countOverlappingPairsOther(unsigned a, unsigned b)
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
							(KeyValuePair *)mortonCodesAndAabbIndices);							
}

void CudaBroadphase::prefixSumPairCounts()
{
    void * scanInput = m_pairCounts->bufferOnDevice();
    void * scanResult = m_pairStart->bufferOnDevice();
    void * scanIntermediate = m_scanIntermediate->bufferOnDevice();
    scanExclusive((uint *)scanResult, (uint *)scanInput, (uint *)scanIntermediate, m_scanBufferLength / 1024, 1024);
}

void CudaBroadphase::writeOverlappingPairs(unsigned a, unsigned b)
{
    if(a==b) writeOverlappingPairsSelf(a);
    else writeOverlappingPairsOther(a, b);
}

void CudaBroadphase::writeOverlappingPairsSelf(unsigned a)
{
    uint * counts = (uint *)m_pairCounts->bufferOnDevice();
	counts += m_objectStart[a];
	
	uint * starts = (uint *)m_pairStart->bufferOnDevice();
	starts += m_objectStart[a];

// only for tetrahedron system	
	CudaTetrahedronSystem * query = static_cast<CudaTetrahedronSystem *>(m_objects[a]);
	CudaLinearBvh * tree = m_objects[a];
	
	void * boxes = (Aabb *)query->leafAabbs();
	const unsigned numBoxes = query->numLeafNodes();
	void * exclusionInd = query->deviceVicinityInd();
	void * exclusionStart = query->deviceVicinityStart();
	
	void * rootNodeIndex = tree->rootNodeIndex();
	void * internalNodeChildIndex = tree->internalNodeChildIndices();
	void * internalNodeAabbs = tree->internalNodeAabbs();
	void * leafNodeAabbs = tree->leafAabbs();
	void * mortonCodesAndAabbIndices = tree->leafHash();
	
	void * cache = m_pairCache->bufferOnDevice();
	
	broadphaseWritePairCacheSelfCollideExclusion((uint2 *)cache, starts, counts, 
	                         (Aabb *)boxes, numBoxes,
							(int *)rootNodeIndex, 
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices,
							a,
							(uint *)exclusionInd,
							(uint *)exclusionStart);
}

void CudaBroadphase::writeOverlappingPairsOther(unsigned a, unsigned b)
{
    uint * counts = (uint *)m_pairCounts->bufferOnDevice();
	counts += m_objectStart[a];
	
	uint * starts = (uint *)m_pairStart->bufferOnDevice();
	starts += m_objectStart[a];
	
	CudaLinearBvh * query = m_objects[a];
	CudaLinearBvh * tree = m_objects[b];
	
	void * boxes = (Aabb *)query->leafAabbs();
	const unsigned numBoxes = query->numLeafNodes();
	
	void * rootNodeIndex = tree->rootNodeIndex();
	void * internalNodeChildIndex = tree->internalNodeChildIndices();
	void * internalNodeAabbs = tree->internalNodeAabbs();
	void * leafNodeAabbs = tree->leafAabbs();
	void * mortonCodesAndAabbIndices = tree->leafHash();
	
	void * cache = m_pairCache->bufferOnDevice();
	
	broadphaseWritePairCache((uint2 *)cache, starts, counts, 
	                         (Aabb *)boxes, numBoxes,
							(int *)rootNodeIndex, 
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices,
							a, b);
}

void CudaBroadphase::getBoxes(BaseBuffer * dst)
{
    char * hbox = (char *)dst->data();
    unsigned i;
    for(i = 0; i<m_numObjects; i++) {
        const unsigned numBoxes = m_objects[i]->numLeafNodes();
        m_objects[i]->getLeafAabbsAt(hbox);
		hbox += numBoxes * 24;
	}
}

unsigned CudaBroadphase::getScanResult(CUDABuffer * counts, CUDABuffer * sums, unsigned n)
{
    unsigned a, b;
    counts->deviceToHost(&a, 4*n, 4);
    sums->deviceToHost(&b, 4*n, 4);
    return a + b;
}
