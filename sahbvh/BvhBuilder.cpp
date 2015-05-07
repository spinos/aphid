/*
 *  BvhBuilder.cpp
 *  testsah
 *
 *  Created by jian zhang on 5/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "BvhBuilder.h"
#include <CudaReduction.h>
#include <CudaLinearBvh.h>
#include <radixsort_implement.h>
#include <createBvh_implement.h>
#include <CUDABuffer.h>
#include <CudaBase.h>
#include <CudaScan.h>

BvhBuilder::BvhBuilder() 
{
	m_findMaxDistance = new CudaReduction;
	m_sortIntermediate = new CUDABuffer;
	m_findPrefixSum = new CudaScan;
}

BvhBuilder::~BvhBuilder() 
{
	delete m_findMaxDistance;
	delete m_sortIntermediate;
}

void BvhBuilder::initOnDevice()
{
	m_findMaxDistance->initOnDevice();
}

CudaReduction * BvhBuilder::reducer()
{ return m_findMaxDistance; }

CudaScan * BvhBuilder::scanner()
{ return m_findPrefixSum; }

void BvhBuilder::createSortAndScanBuf(unsigned n)
{ 
	m_sortIntermediate->create((nextPow2(n) * sizeof(KeyValuePair))); 
	m_findPrefixSum->create(CudaScan::getScanBufferLength(n));
}

void BvhBuilder::build(CudaLinearBvh * bvh) {}

void BvhBuilder::sort(void * odata, unsigned nelem, unsigned nbits)
{
	RadixSort((KeyValuePair *)odata, (KeyValuePair *)sortIntermediate(), nextPow2(nelem), nbits);
}

void BvhBuilder::computeMortionHash(void * mortonCode,
									void * primitiveAabbs, 
									unsigned numPrimitives,
									float * bounding)
{
// find bounding box of all leaf aabb
	reducer()->minMaxBox<Aabb, float3>((Aabb *)bounding, (float3 *)primitiveAabbs, numPrimitives * 2);
	
#if PRINT_BOUND	
    std::cout<<" bvh builder bound (("<<bounding[0]<<" "<<bounding[1]<<" "<<bounding[2];
    std::cout<<"),("<<bounding[3]<<" "<<bounding[4]<<" "<<bounding[5]<<"))";
#endif

	CudaBase::CheckCudaError("finding bvh bounding");
	
// morton curve ordering 
	const unsigned sortLength = nextPow2(numPrimitives);
	
	bvhCalculateLeafHash((KeyValuePair *)mortonCode, (Aabb *)primitiveAabbs, numPrimitives, sortLength,
	    (Aabb *)reducer()->resultOnDevice());
		
	CudaBase::CheckCudaError("calc morton key");
}

void * BvhBuilder::sortIntermediate()
{ return m_sortIntermediate->bufferOnDevice(); }
