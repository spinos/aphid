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

BvhBuilder::BvhBuilder() 
{
	m_findMaxDistance = new CudaReduction;
	m_sortIntermediate = new CUDABuffer;
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

void BvhBuilder::createSortBuf(unsigned n)
{ m_sortIntermediate->create((nextPow2(n) * sizeof(KeyValuePair))); }

void * BvhBuilder::sortIntermediateBuf()
{ return m_sortIntermediate->bufferOnDevice(); }

void BvhBuilder::build(CudaLinearBvh * bvh) {}
