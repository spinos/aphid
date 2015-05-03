/*
 *  LBvhBuilder.cpp
 *  testsah
 *
 *  Created by jian zhang on 5/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "LBvhBuilder.h"
#include <CudaReduction.h>
#include <CudaLinearBvh.h>
#include <radixsort_implement.h>
#include <createBvh_implement.h>
#include <CudaBase.h>

LBvhBuilder::LBvhBuilder() 
{
	m_internalNodeCommonPrefixValues = new CUDABuffer;
	m_internalNodeCommonPrefixLengths = new CUDABuffer;
}

LBvhBuilder::~LBvhBuilder() 
{
	delete m_internalNodeCommonPrefixValues;
	delete m_internalNodeCommonPrefixLengths;
}

void LBvhBuilder::initOnDevice()
{
	BvhBuilder::initOnDevice();
}

void LBvhBuilder::build(CudaLinearBvh * bvh)
{
// create data
	const unsigned nin = bvh->numInternalNodes();
	m_internalNodeCommonPrefixValues->create(nin * sizeof(uint64));
	m_internalNodeCommonPrefixLengths->create(nin * sizeof(int));
	
// find bounding box of all leaf aabb
	const unsigned nl = bvh->numLeafNodes();
	
	Aabb bounding;
	reducer()->minMaxBox<Aabb, float3>(&bounding, (float3 *)bvh->leafAabbs(), nl * 2);
#if PRINT_BOUND	
    std::cout<<" bvh builder bound (("<<bounding.low.x<<" "<<bounding.low.y<<" "<<bounding.low.z;
    std::cout<<"),("<<bounding.high.x<<" "<<bounding.high.y<<" "<<bounding.high.z<<"))";
#endif
	CudaBase::CheckCudaError("finding bvh bounding");

// morton curve ordering 
	const unsigned sortLength = nextPow2(nl);
	
	void * dst = bvh->leafHash0();
	void * src = bvh->leafAabbs();
	bvhCalculateLeafHash((KeyValuePair *)dst, (Aabb *)src, nl, sortLength,
	    (Aabb *)reducer()->resultOnDevice());
		
	CudaBase::CheckCudaError("calc morton key");
	
	RadixSort((KeyValuePair *)dst, (KeyValuePair *)bvh->leafHash1(), sortLength, 32);
	
	void * morton = bvh->leafHash0();
	void * commonPrefix = m_internalNodeCommonPrefixValues->bufferOnDevice();
	void * commonPrefixLengths = m_internalNodeCommonPrefixLengths->bufferOnDevice();
	
	bvhComputeAdjacentPairCommonPrefix((KeyValuePair *)morton,
										(uint64 *)commonPrefix,
										(int *)commonPrefixLengths,
										nin);
	
	void * leafNodeParentIndex = bvh->leafNodeParentIndices();
	void * internalNodeChildIndex = bvh->internalNodeChildIndices();
	
	bvhConnectLeafNodesToInternalTree((int *)commonPrefixLengths, 
								(int *)leafNodeParentIndex,
								(int2 *)internalNodeChildIndex, 
								nl);
								
	void * internalNodeParentIndex = bvh->internalNodeParentIndices();
	void * rootInd = bvh->rootNodeIndex();
	
	bvhConnectInternalTreeNodes((uint64 *)commonPrefix, (int *)commonPrefixLengths,
								(int2 *)internalNodeChildIndex,
								(int *)internalNodeParentIndex,
								(int *)rootInd,
								nin);
								
	void * distanceFromRoot = bvh->distanceInternalNodeFromRoot();
	
	bvhFindDistanceFromRoot((int *)rootInd, (int *)internalNodeParentIndex,
							(int *)distanceFromRoot, 
							nin);
				
	int maxDistance = -1;
	reducer()->max<int>(maxDistance, (int *)distanceFromRoot, nin);
	
#if PRINT_BVH_MAXLEVEL
	std::cout<<" bvh builder max level "<<maxDistance<<"\n";
#endif
	if(maxDistance < 0)
		CudaBase::CheckCudaError("finding bvh max level");
		
	void * boxes =  bvh->leafHash0();
	void * leafNodeAabbs = bvh->leafAabbs();
	void * internalNodeAabbs = bvh->internalNodeAabbs();
	void * maxChildElement = bvh->maxChildElementIndices();
	
	for(int distance = maxDistance; distance >= 0; --distance) {		
		bvhFormInternalNodeAabbsAtDistance((int *)distanceFromRoot, (KeyValuePair *)boxes,
											(int2 *)internalNodeChildIndex,
											(Aabb *)leafNodeAabbs, 
											(Aabb *)internalNodeAabbs,
											(int *)maxChildElement,
											maxDistance, distance, 
											bvh->numInternalNodes());
											
		CudaBase::CheckCudaError("bvh builder form internal aabb iterative");
	}
}
