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
	const unsigned nl = bvh->numPrimitives();
	createSortAndScanBuf(nl);
	
	const unsigned nin = bvh->numInternalNodes();
	m_internalNodeCommonPrefixValues->create(nin * sizeof(uint64));
	m_internalNodeCommonPrefixLengths->create(nin * sizeof(int));
	
	float bounding[6];
	computeMortionHash(bvh->leafHash(), bvh->leafAabbs(), nl, bounding);
	
	sort(bvh->leafHash(), nl, 32);
	
	void * morton = bvh->leafHash();
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
		
	void * boxes =  bvh->leafHash();
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
