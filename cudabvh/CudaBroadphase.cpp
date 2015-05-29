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
#include <CudaTetrahedronSystem.h>
#include <CudaDbgLog.h>

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
	m_pairCounts->create(m_scanBufferLength * 4);
	m_pairStart->create(m_scanBufferLength * 4);
	m_pairWriteLocation->create(m_scanBufferLength * 4);
	m_scanIntermediate->create(m_scanBufferLength);
	
	// DynGlobal::BvhStackedNumThreads = CudaBase::LimitNThreadPerBlock(22, 512+64+16);
	DynGlobal::BvhStackedNumThreads = 32;
    std::cout<<" bvh stack tpb "<<DynGlobal::BvhStackedNumThreads<<"\n";
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

	/*
	bphlg.writeUInt(m_pairCounts,
         m_numBoxes,
                "overlapping_counts", CudaDbgLog::FAlways);

    bphlg.writeUInt(m_pairStart,
         m_numBoxes,
                "overlapping_offsets", CudaDbgLog::FAlways);

	
	std::cout<<" overlapping pair cache length "<<m_pairCacheLength<<"\n";
	*/
	setWriteLocation();
	/*
	bphlg.writeUInt(m_pairWriteLocation,
         m_numBoxes,
                "overlapping_write_location0", CudaDbgLog::FAlways);
	*/
	m_pairCache->create(m_pairCacheLength * 8);
	
	void * cache = m_pairCache->bufferOnDevice();
	broadphaseResetPairCache((uint2 *)cache, m_pairCacheLength);
	
	for(j = 0; j<m_numObjects; j++) {
		for(i = j; i<m_numObjects; i++) {
			writeOverlappingPairs(j, i);
		}
	}
	/*
	bphlg.writeUInt(m_pairWriteLocation,
         m_numBoxes,
                "overlapping_write_location1", CudaDbgLog::FAlways);
    
    bphlg.writeHash(m_pairCache,
         m_pairCacheLength,
                "overlapping_pairs", CudaDbgLog::FAlways);
	*/	
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
    uint * counts = (uint *)m_pairCounts->bufferOnDevice();
	counts += m_objectStart[a];

// only for tetrahedron system
	CudaTetrahedronSystem * query = static_cast<CudaTetrahedronSystem *>(m_objects[a]);
	CudaLinearBvh * tree = m_objects[a];
	
	void * boxes = (Aabb *)query->leafAabbs();
	const unsigned numBoxes = query->numLeafNodes();
	void * exclusionInd = query->deviceVicinityInd();
	void * exclusionStart = query->deviceVicinityStart();
	
	void * internalNodeChildIndex = tree->internalNodeChildIndices();
	void * internalNodeAabbs = tree->internalNodeAabbs();
	void * leafNodeAabbs = tree->primitiveAabb();
	void * mortonCodesAndAabbIndices = tree->primitiveHash();
    
	cubroadphase::countPairsSelfCollideExclS(counts, (Aabb *)boxes, numBoxes,
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices,
							(uint *)exclusionInd,
							(uint *)exclusionStart,
							DynGlobal::BvhStackedNumThreads);
	CudaBase::CheckCudaError("broadphase count pairs smem");
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
	void * leafNodeAabbs = tree->primitiveAabb();
	void * mortonCodesAndAabbIndices = tree->primitiveHash();
	
	broadphaseComputePairCounts(counts, (Aabb *)boxes, numBoxes,
							(int *)rootNodeIndex, 
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices);							
}

void CudaBroadphase::setWriteLocation()
{
    void * dst = m_pairWriteLocation->bufferOnDevice();
    void * src = m_pairStart->bufferOnDevice();
    cubroadphase::writeLocation((uint *)dst, (uint *)src, m_scanBufferLength);
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
	
	uint * location = (uint *)m_pairWriteLocation->bufferOnDevice();
	location += m_objectStart[a];

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
	
	cubroadphase::writePairCacheSelfCollideExclS((uint2 *)cache, location, starts, counts, 
	                         (Aabb *)boxes, numBoxes,
							(int *)rootNodeIndex, 
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices,
							a,
							(uint *)exclusionInd,
							(uint *)exclusionStart,
							DynGlobal::BvhStackedNumThreads);
	CudaBase::CheckCudaError("broadphase write pairs smem");
}

void CudaBroadphase::writeOverlappingPairsOther(unsigned a, unsigned b)
{
    uint * counts = (uint *)m_pairCounts->bufferOnDevice();
	counts += m_objectStart[a];
	
	uint * starts = (uint *)m_pairStart->bufferOnDevice();
	starts += m_objectStart[a];
	
	uint * location = (uint *)m_pairWriteLocation->bufferOnDevice();
	location += m_objectStart[a];
	
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
	
	cuBroadphase_writePairCache((uint2 *)cache, location, starts, counts, 
	                         (Aabb *)boxes, numBoxes,
							(int *)rootNodeIndex, 
							(int2 *)internalNodeChildIndex, 
							(Aabb *)internalNodeAabbs, 
							(Aabb *)leafNodeAabbs,
							(KeyValuePair *)mortonCodesAndAabbIndices,
							a, b);
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