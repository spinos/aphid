/*
 *  CudaBroadphase.cpp
 *  cudabvh
 *
 *  Created by jian zhang on 2/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <CUDABuffer.h>
#include "CudaLinearBvh.h"
#include "CudaBroadphase.h"
#include "broadphase_implement.h"
CudaBroadphase::CudaBroadphase() 
{
	m_numObjects = 0;
	m_pairCounts = new CUDABuffer;
}

CudaBroadphase::~CudaBroadphase() {}

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
	m_pairCounts->create(m_numBoxes * 4);
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
}

void CudaBroadphase::resetPairCounts()
{
	broadphaseResetPairCounts((uint *)m_pairCounts->bufferOnDevice(), m_numBoxes);
}

void CudaBroadphase::countOverlappingPairs(unsigned a, unsigned b)
{
	uint * counts = (uint *)m_pairCounts->bufferOnDevice();
	counts += m_objectStart[a];
}
