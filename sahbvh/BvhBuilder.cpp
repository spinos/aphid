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
#include <CudaBase.h>

BvhBuilder::BvhBuilder() 
{
	m_findMaxDistance = new CudaReduction;
}

BvhBuilder::~BvhBuilder() 
{
	delete m_findMaxDistance;
}

void BvhBuilder::initOnDevice()
{
	m_findMaxDistance->initOnDevice();
}

CudaReduction * BvhBuilder::reducer()
{ return m_findMaxDistance; }

void BvhBuilder::build(CudaLinearBvh * bvh) {}
