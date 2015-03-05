/*
 *  CudaNarrowphase.cpp
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "CudaNarrowphase.h"
#include <CudaTetrahedronSystem.h>
#include <CUDABuffer.h>
#include "narrowphase_implement.h"

CudaNarrowphase::CudaNarrowphase() 
{
    m_numObjects = 0;
	m_numContacts = 0;
    m_pos = new CUDABuffer;
    m_vel = new CUDABuffer;
    m_ind = new CUDABuffer;
	m_pointCacheLoc = new CUDABuffer;
	m_indexCacheLoc = new CUDABuffer;
	m_separateAxis = new CUDABuffer;
	m_pointA = new CUDABuffer;
	m_pointB = new CUDABuffer;
}

CudaNarrowphase::~CudaNarrowphase() {}

void CudaNarrowphase::getSeparateAxis(BaseBuffer * dst)
{ m_separateAxis->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaNarrowphase::getPointA(BaseBuffer * dst)
{ m_pointA->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaNarrowphase::getPointB(BaseBuffer * dst)
{ m_pointB->deviceToHost(dst->data(), dst->bufferSize()); }

const unsigned CudaNarrowphase::numContacts() const
{ return m_numContacts; }

void CudaNarrowphase::addTetrahedronSystem(CudaTetrahedronSystem * tetra)
{
    if(m_numObjects==CUDANARROWPHASE_MAX_NUMOBJECTS) return;
    m_objects[m_numObjects] = tetra;
    m_numObjects++;
}

void CudaNarrowphase::initOnDevice()
{
	if(m_numObjects < 1) return;
	m_objectPointStart[0] = 0;
	m_objectIndexStart[0] = 0;
	m_numElements = 0;
	m_numPoints = 0;
	unsigned i;
	for(i = 0; i<m_numObjects; i++) {
		m_numElements += m_objects[i]->numTetradedrons();
		m_numPoints += m_objects[i]->numPoints();
		if(i<m_numObjects-1) {
		    m_objectPointStart[i+1] = m_numPoints;
			m_objectIndexStart[i+1] = m_numElements;
		}
	}
	
	m_pos->create(m_numPoints * 12);
	m_vel->create(m_numPoints * 12);
	m_ind->create(m_numElements * 4 * 4);
	
	m_pointCacheLoc->create(CUDANARROWPHASE_MAX_NUMOBJECTS * 4);
	m_indexCacheLoc->create(CUDANARROWPHASE_MAX_NUMOBJECTS * 4);
	
	m_pointCacheLoc->hostToDevice(&m_objectPointStart[0]);
	m_indexCacheLoc->hostToDevice(&m_objectIndexStart[0]);
}

void CudaNarrowphase::computeContacts(CUDABuffer * overlappingPairBuf, unsigned numOverlappingPairs)
{
    if(numOverlappingPairs < 1) return;
    
    unsigned i;
    for(i = 0; i<m_numObjects; i++) {
	    writeObjectCache(m_objects[i], 
	        m_objectPointStart[i], m_objectIndexStart[i]);
	}
	
	m_separateAxis->create(numOverlappingPairs * 16);
	m_pointA->create(numOverlappingPairs * 12);
	m_pointB->create(numOverlappingPairs * 12);
	m_numContacts = numOverlappingPairs;
	computeSeparateAxis(overlappingPairBuf, numOverlappingPairs);
}

void CudaNarrowphase::writeObjectCache(CudaTetrahedronSystem * tetra, 
	        unsigned pointAt, unsigned indexAt)
{
    char * dstPos = (char *)m_pos->bufferOnDevice();
    char * dstVel = (char *)m_vel->bufferOnDevice();
    char * dstInd = (char *)m_ind->bufferOnDevice();
    dstPos += pointAt * 12;
    dstVel += pointAt * 12;
    dstInd += indexAt * 16;

    narrowphaseWriteObjectToCache((float3 *)dstPos,
        (float3 *)dstVel,
        (uint4 *)dstInd,
        (float3 *)tetra->deviceX(),
        (float3 *)tetra->deviceV(),
        (uint4 *)tetra->deviceTretradhedronIndices(),
        tetra->numPoints(),
		tetra->numTetradedrons());
}

void CudaNarrowphase::computeSeparateAxis(CUDABuffer * overlappingPairBuf, unsigned numOverlappingPairs)
{
	void * dstSA = m_separateAxis->bufferOnDevice();
	void * dstPA = m_pointA->bufferOnDevice();
	void * dstPB = m_pointB->bufferOnDevice();
	void * pairs = overlappingPairBuf->bufferOnDevice();
	void * pos = m_pos->bufferOnDevice();
	void * vel = m_vel->bufferOnDevice();
	void * ind = m_ind->bufferOnDevice();
	narrowphaseComputeSeparateAxis((float4 *)dstSA,
	    (float3 *)dstPA,
	    (float3 *)dstPB,
		(uint2 *)pairs,
		(float3 *)pos,
		(float3 *)vel,
		(uint4 *)ind,
		(uint *)m_pointCacheLoc->bufferOnDevice(),
		(uint *)m_indexCacheLoc->bufferOnDevice(),
		numOverlappingPairs);
}
