/*
 *  CudaLinearBvh.cpp
 *  cudabvh
 *
 *  Created by jian zhang on 1/15/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <CudaBase.h>
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include "BvhTriangleMesh.h"
#include "CudaLinearBvh.h"
#include <radixsort_implement.h>
#include "createBvh_implement.h"
#include "CudaReduction.h"
#include "bvh_dbg.h"

CudaLinearBvh::CudaLinearBvh() 
{ 
	m_numLeafNodes = 0; 
	m_leafAabbs = new CUDABuffer;
	m_internalNodeAabbs = new CUDABuffer;
	m_leafHash[0] = new CUDABuffer;
	m_leafHash[1] = new CUDABuffer;
	m_internalNodeCommonPrefixValues = new CUDABuffer;
	m_internalNodeCommonPrefixLengths = new CUDABuffer;
	m_leafNodeParentIndices = new CUDABuffer;
	m_internalNodeChildIndices = new CUDABuffer;
	m_internalNodeParentIndices = new CUDABuffer;
	m_rootNodeIndexOnDevice = new CUDABuffer;
	m_distanceInternalNodeFromRoot = new CUDABuffer;
	m_maxChildElementIndices = new CUDABuffer;
	m_findMaxDistance = new CudaReduction;
#if DRAW_BVH_HASH
	m_hostLeafHash = new BaseBuffer;
	m_hostLeafBox = new BaseBuffer;
#endif
#if DRAW_BVH_HIERARCHY
	m_hostInternalAabb = new BaseBuffer;
#endif
}

CudaLinearBvh::~CudaLinearBvh() 
{
	delete m_leafAabbs;
	delete m_internalNodeAabbs;
	delete m_leafHash[0];
	delete m_leafHash[1];
	delete m_internalNodeCommonPrefixValues;
	delete m_internalNodeCommonPrefixLengths;
	delete m_leafNodeParentIndices;
	delete m_internalNodeChildIndices;
	delete m_internalNodeParentIndices;
	delete m_rootNodeIndexOnDevice;
	delete m_distanceInternalNodeFromRoot;
	delete m_maxChildElementIndices;
	delete m_findMaxDistance;
}

void CudaLinearBvh::setNumLeafNodes(unsigned n)
{ m_numLeafNodes = n; }

void CudaLinearBvh::initOnDevice()
{
	m_leafAabbs->create(numLeafNodes() * sizeof(Aabb));
// assume numInternalNodes() >> ReduceMaxBlocks
	m_internalNodeAabbs->create(numInternalNodes() * sizeof(Aabb));
	
	m_leafHash[0]->create(nextPow2(numLeafNodes()) * sizeof(KeyValuePair));
	m_leafHash[1]->create(nextPow2(numLeafNodes()) * sizeof(KeyValuePair));
	
	m_internalNodeCommonPrefixValues->create(numInternalNodes() * sizeof(uint64));
	m_internalNodeCommonPrefixLengths->create(numInternalNodes() * sizeof(int));
	
	m_leafNodeParentIndices->create(numLeafNodes() * sizeof(int));
	m_internalNodeChildIndices->create(numInternalNodes() * sizeof(int2));
	m_internalNodeParentIndices->create(numInternalNodes() * sizeof(int));
	m_rootNodeIndexOnDevice->create(sizeof(int));
	m_distanceInternalNodeFromRoot->create(numInternalNodes() * sizeof(int));
	m_maxChildElementIndices->create(numInternalNodes() * sizeof(int));
	
	m_findMaxDistance->initOnDevice();

#if DRAW_BVH_HASH
	m_hostLeafBox->create(numLeafNodes() * sizeof(Aabb));
	m_hostLeafHash->create(nextPow2(numLeafNodes()) * sizeof(KeyValuePair));
#endif

}

const unsigned CudaLinearBvh::numLeafNodes() const 
{ return m_numLeafNodes; }

const unsigned CudaLinearBvh::numInternalNodes() const 
{ return numLeafNodes() - 1; }

void CudaLinearBvh::getRootNodeIndex(int * dst)
{ m_rootNodeIndexOnDevice->deviceToHost((void *)dst, m_rootNodeIndexOnDevice->bufferSize()); }

void CudaLinearBvh::getInternalAabbs(BaseBuffer * dst)
{ m_internalNodeAabbs->deviceToHost(dst->data(), m_internalNodeAabbs->bufferSize()); }

void CudaLinearBvh::getInternalDistances(BaseBuffer * dst)
{ m_distanceInternalNodeFromRoot->deviceToHost(dst->data(), m_distanceInternalNodeFromRoot->bufferSize()); }

void CudaLinearBvh::getInternalChildIndex(BaseBuffer * dst)
{ m_internalNodeChildIndices->deviceToHost(dst->data(), m_internalNodeChildIndices->bufferSize()); }

void * CudaLinearBvh::rootNodeIndex()
{ return m_rootNodeIndexOnDevice->bufferOnDevice(); }

void * CudaLinearBvh::internalNodeChildIndices()
{ return m_internalNodeChildIndices->bufferOnDevice(); }

void * CudaLinearBvh::internalNodeAabbs()
{ return m_internalNodeAabbs->bufferOnDevice(); }

void * CudaLinearBvh::internalNodeChildLimit()
{ return m_maxChildElementIndices->bufferOnDevice(); }

void * CudaLinearBvh::leafAabbs()
{ return m_leafAabbs->bufferOnDevice(); }

void CudaLinearBvh::getLeafAabbsAt(char * dst)
{ m_leafAabbs->deviceToHost(dst, 0, m_leafAabbs->bufferSize()); }

void * CudaLinearBvh::leafHash()
{ return m_leafHash[0]->bufferOnDevice(); }
 
void CudaLinearBvh::update()
{
	computeAabb();
	computeAndSortLeafHash();
	buildInternalTree();
}

void CudaLinearBvh::computeAabb()
{
	reducer()->minMaxBox<Aabb, float3>(&m_bounding, (float3 *)leafAabbs(), numLeafNodes() * 2);
#if PRINT_BOUND
    std::cout<<" bound (("<<m_bounding.low.x<<" "<<m_bounding.low.y<<" "<<m_bounding.low.z;
    std::cout<<"),("<<m_bounding.high.x<<" "<<m_bounding.high.y<<" "<<m_bounding.high.z<<"))";
#endif
}

void CudaLinearBvh::computeAndSortLeafHash()
{
	void * dst = m_leafHash[0]->bufferOnDevice();
	void * src = leafAabbs();
	bvhCalculateLeafHash((KeyValuePair *)dst, (Aabb *)src, numLeafNodes(), nextPow2(numLeafNodes()),
	    m_bounding);
	void * tmp = m_leafHash[1]->bufferOnDevice();
	RadixSort((KeyValuePair *)dst, (KeyValuePair *)tmp, nextPow2(numLeafNodes()), 32);
}

void CudaLinearBvh::buildInternalTree()
{
	void * morton = m_leafHash[0]->bufferOnDevice();
	void * commonPrefix = m_internalNodeCommonPrefixValues->bufferOnDevice();
	void * commonPrefixLengths = m_internalNodeCommonPrefixLengths->bufferOnDevice();
	
	bvhComputeAdjacentPairCommonPrefix((KeyValuePair *)morton,
										(uint64 *)commonPrefix,
										(int *)commonPrefixLengths,
										numInternalNodes());
	
	void * leafNodeParentIndex = m_leafNodeParentIndices->bufferOnDevice();
	void * internalNodeChildIndex = m_internalNodeChildIndices->bufferOnDevice();
	
	bvhConnectLeafNodesToInternalTree((int *)commonPrefixLengths, 
								(int *)leafNodeParentIndex,
								(int2 *)internalNodeChildIndex, 
								numLeafNodes());
								
	void * internalNodeParentIndex = m_internalNodeParentIndices->bufferOnDevice();
	void * rootInd = m_rootNodeIndexOnDevice->bufferOnDevice();
	bvhConnectInternalTreeNodes((uint64 *)commonPrefix, (int *)commonPrefixLengths,
								(int2 *)internalNodeChildIndex,
								(int *)internalNodeParentIndex,
								(int *)rootInd,
								numInternalNodes());
	
	void * distanceFromRoot = m_distanceInternalNodeFromRoot->bufferOnDevice();
	bvhFindDistanceFromRoot((int *)rootInd, (int *)internalNodeParentIndex,
							(int *)distanceFromRoot, 
							numInternalNodes());
				
	int maxDistance = -1;
	m_findMaxDistance->max<int>(maxDistance, (int *)m_distanceInternalNodeFromRoot->bufferOnDevice(), numInternalNodes());
#if PRINT_BVH_MAXLEVEL
	std::cout<<" bvh max level "<<maxDistance<<"\n";
#endif
	if(maxDistance < 0)
		CudaBase::CheckCudaError("finding bvh max level");
	
	formInternalTreeAabbsIterative(maxDistance);
}

void CudaLinearBvh::formInternalTreeAabbsIterative(int maxDistance)
{
	void * distances = m_distanceInternalNodeFromRoot->bufferOnDevice();
	void * boxes = m_leafHash[0]->bufferOnDevice();
	void * internalNodeChildIndex = m_internalNodeChildIndices->bufferOnDevice();
	void * leafNodeAabbs = m_leafAabbs->bufferOnDevice();
	void * internalNodeAabbs = m_internalNodeAabbs->bufferOnDevice();
	void * maxChildElement = m_maxChildElementIndices->bufferOnDevice();
	
	for(int distanceFromRoot = maxDistance; distanceFromRoot >= 0; --distanceFromRoot) {		
		bvhFormInternalNodeAabbsAtDistance((int *)distances, (KeyValuePair *)boxes,
											(int2 *)internalNodeChildIndex,
											(Aabb *)leafNodeAabbs, 
											(Aabb *)internalNodeAabbs,
											(int *)maxChildElement,
											maxDistance, distanceFromRoot, 
											numInternalNodes());
											
		CudaBase::CheckCudaError("bvh form internal aabb iterative");
	}
}

const unsigned CudaLinearBvh::usedMemory() const
{
	unsigned n = m_leafAabbs->bufferSize();
	n += m_internalNodeAabbs->bufferSize();
	n += m_leafHash[0]->bufferSize() * 2;
	n += m_internalNodeCommonPrefixValues->bufferSize();
	n += m_internalNodeCommonPrefixLengths->bufferSize();
	n += m_leafNodeParentIndices->bufferSize();
	n += m_internalNodeChildIndices->bufferSize();
	n += m_internalNodeParentIndices->bufferSize();
	n += m_rootNodeIndexOnDevice->bufferSize();
    n += m_distanceInternalNodeFromRoot->bufferSize();
	return n;
}

#if DRAW_BVH_HASH
void * CudaLinearBvh::hostLeafHash()
{ return m_hostLeafHash->data(); }

void * CudaLinearBvh::hostLeafBox()
{ return m_hostLeafBox->data(); }
#endif

void CudaLinearBvh::sendDbgToHost()
{
#if DRAW_BVH_HASH
	m_leafAabbs->deviceToHost(m_hostLeafBox->data(), m_leafAabbs->bufferSize());
	m_leafHash[0]->deviceToHost(m_hostLeafHash->data(), m_leafHash[0]->bufferSize());
#endif
}

#if DRAW_BVH_HIERARCHY
void * CudaLinearBvh::hostInternalAabb()
{ return m_hostInternalAabb->data(); }
#endif

CudaReduction * CudaLinearBvh::reducer()
{ return m_findMaxDistance; }

float * CudaLinearBvh::aabbPtr()
{ return &m_bounding.low.x; }

const Aabb CudaLinearBvh::getAabb() const
{ return m_bounding; }
//:~