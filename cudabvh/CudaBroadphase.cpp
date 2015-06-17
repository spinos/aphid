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
#include <CudaScan.h>
#include <BvhTetrahedronSystem.h>
#include "TetrahedronSystemInterface.h"
#include "OverlappingInterface.h"
#include <CudaDbgLog.h>

//#define DISABLE_INTER_OBJECT_COLLISION
//#define DISABLE_SELF_COLLISION

CudaDbgLog bphlg("broadphase.txt");

CudaBroadphase::CudaBroadphase() 
{
    m_numObjects = 0;
	m_pairCacheLength = 0;
	m_pairCounts = new CUDABuffer;
	m_pairStart = new CUDABuffer;
	m_pairWriteLocation = new CUDABuffer;
	m_scanIntermediate = new CudaScan;
	m_pairCache = new CUDABuffer;
#if DRAW_BPH_PAIRS	
	m_hostPairCache = new BaseBuffer;
	m_hostAabb = new BaseBuffer;
#endif
}

CudaBroadphase::~CudaBroadphase() 
{
    delete m_pairCounts;
    delete m_pairStart;
    delete m_pairWriteLocation;
    delete m_scanIntermediate;
    delete m_pairCache;
    
#if DRAW_BPH_PAIRS
	delete m_hostPairCache;
	delete m_hostAabb;
#endif
}

const unsigned CudaBroadphase::numBoxes() const
{ return m_numBoxes; }

const unsigned CudaBroadphase::pairCacheLength() const
{ return m_pairCacheLength; }

const unsigned CudaBroadphase::numOverlappingPairs() const
{ return m_pairCacheLength; }

const unsigned CudaBroadphase::objectStart(unsigned ind) const
{ return m_objectStart[ind]; }

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
	
	m_scanBufferLength = CudaScan::getScanBufferLength(m_numBoxes);
    std::cout<<" broadphase scan buf length "<<m_scanBufferLength<<"\n";
	m_pairCounts->create(m_scanBufferLength * 4);
	m_pairStart->create(m_scanBufferLength * 4);
	m_pairWriteLocation->create(m_scanBufferLength * 4);
	m_scanIntermediate->create(m_scanBufferLength);
}

void CudaBroadphase::computeOverlappingPairs()
{
#if DISABLE_COLLISION_DETECTION
	return;
#endif	
	if(m_numObjects < 1) return;
	unsigned i, j;
	
	resetPairCounts();
	
	for(j = 0; j<m_numObjects; j++) {
		for(i = j; i<m_numObjects; i++) {
			countOverlappingPairs(j, i);
		}
	}
	
	m_pairCacheLength = m_scanIntermediate->prefixSum(m_pairStart, m_pairCounts, m_scanBufferLength);
	
	if(m_pairCacheLength < 1) return;
#if 0
	bphlg.writeUInt(m_pairCounts,
         m_numBoxes,
                "overlapping_counts", CudaDbgLog::FAlways);
#endif
#if 0
    bphlg.writeUInt(m_pairStart,
         m_numBoxes,
                "overlapping_offsets", CudaDbgLog::FAlways);
	std::cout<<" overlapping pair cache length "<<m_pairCacheLength<<"\n";
#endif	
	setWriteLocation();
#if 0	
	bphlg.writeUInt(m_pairWriteLocation,
         m_numBoxes,
                "overlapping_write_location0", CudaDbgLog::FAlways);
#endif
	m_pairCache->create(m_pairCacheLength * 8);
	
	void * cache = m_pairCache->bufferOnDevice();
	broadphaseResetPairCache((uint2 *)cache, m_pairCacheLength);
	
	for(j = 0; j<m_numObjects; j++) {
		for(i = j; i<m_numObjects; i++) {
			writeOverlappingPairs(j, i);
		}
	}
#if 0
	bphlg.writeUInt(m_pairWriteLocation,
         m_numBoxes,
                "overlapping_write_location1", CudaDbgLog::FAlways);
#endif
#if 0    
    bphlg.writeHash(m_pairCache,
         m_pairCacheLength,
                "overlapping_pairs", CudaDbgLog::FAlways);
#endif	
#if DRAW_BPH_PAIRS
	m_hostPairCache->create(m_pairCacheLength * 8);
	m_hostAabb->create(m_numBoxes * sizeof(Aabb));
#endif
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
#ifdef DISABLE_SELF_COLLISION 
    return;
#endif
    uint * counts = (uint *)m_pairCounts->bufferOnDevice();
	counts += m_objectStart[a];

// only for tetrahedron system
	BvhTetrahedronSystem * query = static_cast<BvhTetrahedronSystem *>(m_objects[a]);
	CudaLinearBvh * tree = m_objects[a];
	
	void * boxes = query->leafAabbs();
	void * exclusionInd = query->vicinity();
	
	void * internalNodeChildIndex = tree->internalNodeChildIndices();
	void * internalNodeAabbs = tree->internalNodeAabbs();
	void * leafNodeAabbs = tree->primitiveAabb();
	void * mortonCodesAndAabbIndices = tree->primitiveHash();
    
	bvhoverlap::countPairsSelfCollide(counts, (Aabb *)boxes,
	                        query->numActiveInternalNodes(),
							query->numPrimitives(),
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices,
							(int *)exclusionInd);
	CudaBase::CheckCudaError("broadphase count pairs self-collide");
}

void CudaBroadphase::countOverlappingPairsOther(unsigned a, unsigned b)
{
#ifdef DISABLE_INTER_OBJECT_COLLISION
    return;
#endif
	uint * counts = (uint *)m_pairCounts->bufferOnDevice();
	counts += m_objectStart[a];
	
	CudaLinearBvh * query = m_objects[a];
	CudaLinearBvh * tree = m_objects[b];
	
	void * boxes = query->leafAabbs();
	void * queryInd = query->primitiveHash();
	const unsigned numBoxes = query->numLeafNodes();
	
	void * internalNodeChildIndex = tree->internalNodeChildIndices();
	void * internalNodeAabbs = tree->internalNodeAabbs();
	void * leafNodeAabbs = tree->primitiveAabb();
	void * mortonCodesAndAabbIndices = tree->primitiveHash();
	
	bvhoverlap::countPairs(counts, (Aabb *)boxes, 
	                        (KeyValuePair *)queryInd,
	                         query->numActiveInternalNodes(),
							numBoxes,
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices);	
    CudaBase::CheckCudaError("broadphase count pairs");
}

void CudaBroadphase::setWriteLocation()
{
    void * dst = m_pairWriteLocation->bufferOnDevice();
    void * src = m_pairStart->bufferOnDevice();
    bvhoverlap::writeLocation((uint *)dst, (uint *)src, m_scanBufferLength);
}

void CudaBroadphase::writeOverlappingPairs(unsigned a, unsigned b)
{
    if(a==b) writeOverlappingPairsSelf(a);
    else writeOverlappingPairsOther(a, b);
}

void CudaBroadphase::writeOverlappingPairsSelf(unsigned a)
{
#ifdef DISABLE_SELF_COLLISION 
    return;
#endif
    uint * counts = (uint *)m_pairCounts->bufferOnDevice();
	counts += m_objectStart[a];
	
	uint * starts = (uint *)m_pairStart->bufferOnDevice();
	starts += m_objectStart[a];
	
	uint * location = (uint *)m_pairWriteLocation->bufferOnDevice();
	location += m_objectStart[a];

// only for tetrahedron system	
	BvhTetrahedronSystem * query = static_cast<BvhTetrahedronSystem *>(m_objects[a]);
	CudaLinearBvh * tree = m_objects[a];
	
	void * boxes = query->leafAabbs();
	void * exclusionInd = query->vicinity();
	
	void * internalNodeChildIndex = tree->internalNodeChildIndices();
	void * internalNodeAabbs = tree->internalNodeAabbs();
	void * leafNodeAabbs = tree->leafAabbs();
	void * mortonCodesAndAabbIndices = tree->primitiveHash();
	
	void * cache = m_pairCache->bufferOnDevice();
	
	bvhoverlap::writePairCacheSelfCollide((uint2 *)cache, 
	                            location,
	                            starts,
	                            counts,
	                         (Aabb *)boxes, 
	                         query->numActiveInternalNodes(),
							query->numPrimitives(),
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices,
							a,
							(int *)exclusionInd);
	CudaBase::CheckCudaError("broadphase write pairs self-collide");
}

void CudaBroadphase::writeOverlappingPairsOther(unsigned a, unsigned b)
{
#ifdef DISABLE_INTER_OBJECT_COLLISION
    return;
#endif
    uint * counts = (uint *)m_pairCounts->bufferOnDevice();
	counts += m_objectStart[a];
	
	uint * starts = (uint *)m_pairStart->bufferOnDevice();
	starts += m_objectStart[a];
	
	uint * location = (uint *)m_pairWriteLocation->bufferOnDevice();
	location += m_objectStart[a];
	
	CudaLinearBvh * query = m_objects[a];
	CudaLinearBvh * tree = m_objects[b];
	
	void * boxes = query->leafAabbs();
	void * queryInd = query->primitiveHash();
	
	void * internalNodeChildIndex = tree->internalNodeChildIndices();
	void * internalNodeAabbs = tree->internalNodeAabbs();
	void * leafNodeAabbs = tree->leafAabbs();
	void * mortonCodesAndAabbIndices = tree->leafHash();
	
	void * cache = m_pairCache->bufferOnDevice();
	
	bvhoverlap::writePairCache((uint2 *)cache, 
	                            location, 
	                            starts,
	                            counts,
	                         (Aabb *)boxes, 
	                         (KeyValuePair *)queryInd,
	                            query->numActiveInternalNodes(),
							query->numPrimitives(),
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices,
							a, b);
    CudaBase::CheckCudaError("broadphase write pairs");
}

void CudaBroadphase::sendDbgToHost()
{
#if DRAW_BPH_PAIRS
	if(m_pairCacheLength<1) return;
	m_pairCache->deviceToHost(m_hostPairCache->data());
	char * hbox = (char *)hostAabb();
	unsigned i=0;
    for(; i<m_numObjects; i++) {
        const unsigned nb = m_objects[i]->numLeafNodes();
        m_objects[i]->getLeafAabbsAt(hbox);
		hbox += nb * 24;
	}
#endif
}

#if DRAW_BPH_PAIRS
void * CudaBroadphase::hostPairCache()
{ return m_hostPairCache->data(); }

void * CudaBroadphase::hostAabb()
{ return m_hostAabb->data(); }
#endif