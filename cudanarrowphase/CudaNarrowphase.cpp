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

struct SSimplex {
    float3 p[4];
	float3 pA[4];
	float3 pB[4];
	int dimension;
};

struct ClosestPointTestContext {
    // float3 referencePoint;
    float3 closestPoint;
    float closestDistance;
};

CudaNarrowphase::CudaNarrowphase() 
{
    m_numObjects = 0;
	m_numContacts = 0;
    m_pos = new CUDABuffer;
    m_vel = new CUDABuffer;
    m_ind = new CUDABuffer;
	m_pointCacheLoc = new CUDABuffer;
	m_indexCacheLoc = new CUDABuffer;
	m_coord = new CUDABuffer;
	m_contact = new CUDABuffer;
	
	std::cout<<" size of simplex "<<sizeof(SSimplex)<<" \n";
	std::cout<<" size of ctc "<<sizeof(ClosestPointTestContext)<<" \n";

}

CudaNarrowphase::~CudaNarrowphase() {}

void CudaNarrowphase::getCoord(BaseBuffer * dst)
{ m_coord->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaNarrowphase::getContact(BaseBuffer * dst)
{ m_contact->deviceToHost(dst->data(), dst->bufferSize()); }

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
	m_ind->create(m_numElements * 16); // 4 ints
	
	m_pointCacheLoc->create(CUDANARROWPHASE_MAX_NUMOBJECTS * 4);
	m_indexCacheLoc->create(CUDANARROWPHASE_MAX_NUMOBJECTS * 4);
	
	m_pointCacheLoc->hostToDevice(&m_objectPointStart[0]);
	m_indexCacheLoc->hostToDevice(&m_objectIndexStart[0]);
	
	char * pos = (char *)m_pos->bufferOnDevice();
	char * vel = (char *)m_vel->bufferOnDevice();
	char * ind = (char *)m_ind->bufferOnDevice();
	
	for(i = 0; i<m_numObjects; i++) {
		CudaTetrahedronSystem * curObj = m_objects[i];
		
		curObj->setDeviceXPtr(pos);
		curObj->setDeviceXPtr(vel);
		curObj->setDeviceTretradhedronIndicesPtr(ind);
		
		m_pos->hostToDevice(curObj->hostX(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
		m_vel->hostToDevice(curObj->hostV(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
		m_ind->hostToDevice(curObj->hostTretradhedronIndices(), m_objectIndexStart[i] * 16, curObj->numTetradedrons() * 16);
		
		curObj->initOnDevice();
		
		pos += m_objectPointStart[i] * 12;
		vel += m_objectPointStart[i] * 12;
		ind += m_objectIndexStart[i] * 16;
	}
}

void CudaNarrowphase::computeContacts(CUDABuffer * overlappingPairBuf, unsigned numOverlappingPairs)
{
    if(numOverlappingPairs < 1) return;
	
	m_coord->create(numOverlappingPairs * 16);
	m_contact->create(numOverlappingPairs * 48);
	m_numContacts = numOverlappingPairs;
	computeSeparateAxis(overlappingPairBuf, numOverlappingPairs);
}

void CudaNarrowphase::computeSeparateAxis(CUDABuffer * overlappingPairBuf, unsigned numOverlappingPairs)
{
	void * dstCoord = m_coord->bufferOnDevice();
	void * dstContact = m_contact->bufferOnDevice();
	void * pairs = overlappingPairBuf->bufferOnDevice();
	void * pos = m_pos->bufferOnDevice();
	void * vel = m_vel->bufferOnDevice();
	void * ind = m_ind->bufferOnDevice();
	narrowphaseComputeSeparateAxis((ContactData *)dstContact,
		(uint2 *)pairs,
		(float3 *)pos,
		(float3 *)vel,
		(uint4 *)ind,
		(uint *)m_pointCacheLoc->bufferOnDevice(),
		(uint *)m_indexCacheLoc->bufferOnDevice(),
		numOverlappingPairs);
}
