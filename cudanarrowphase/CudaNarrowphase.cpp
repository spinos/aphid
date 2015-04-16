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
#include "scan_implement.h"
#include <ScanUtil.h>

struct SSimplex {
    float3 p[4];
	float3 pA[4];
	float3 pB[4];
	int dimension;
};

struct ClosestPointTestContext {
    float3 closestPoint;
    float closestDistance;
};

struct SContactData {
    float4 separateAxis;
    float3 localA;
    float3 localB;
    float toi;
};

CudaNarrowphase::CudaNarrowphase() 
{
    m_numObjects = 0;
	m_numPairs = 0;
	m_numContacts = 0;
    m_objectBuf.m_pos = new CUDABuffer;
    m_objectBuf.m_pos0 = new CUDABuffer;
    m_objectBuf.m_vel = new CUDABuffer;
    m_objectBuf.m_mass = new CUDABuffer;
    m_objectBuf.m_ind = new CUDABuffer;
	m_objectBuf.m_pointCacheLoc = new CUDABuffer;
	m_objectBuf.m_indexCacheLoc = new CUDABuffer;
	m_coord = new CUDABuffer;
	m_contact[0] = new CUDABuffer;
	m_contact[1] = new CUDABuffer;
	m_contactPairs = new CUDABuffer;
	m_validCounts = new CUDABuffer;
	m_scanValidContacts[0] = new CUDABuffer;
	m_scanValidContacts[1] = new CUDABuffer;
	std::cout<<" size of simplex "<<sizeof(SSimplex)<<" \n";
	std::cout<<" size of ctc "<<sizeof(ClosestPointTestContext)<<" \n";
	std::cout<<" size of contact "<<sizeof(SContactData)<<" \n";

}

CudaNarrowphase::~CudaNarrowphase() {}

const unsigned CudaNarrowphase::numPoints() const
{ return m_numPoints; }

const unsigned CudaNarrowphase::numElements() const
{ return m_numElements; }

void CudaNarrowphase::getCoord(BaseBuffer * dst)
{ m_coord->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaNarrowphase::getContact0(BaseBuffer * dst)
{ m_contact[0]->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaNarrowphase::getContact(BaseBuffer * dst)
{ m_contact[1]->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaNarrowphase::getContactCounts(BaseBuffer * dst)
{ m_validCounts->deviceToHost(dst->data(), dst->bufferSize()); }

const unsigned CudaNarrowphase::numPairs() const
{ return m_numPairs; }

const unsigned CudaNarrowphase::numContacts() const
{ return m_numContacts; }

void * CudaNarrowphase::contacts0()
{ return m_contact[0]->bufferOnDevice(); }

void * CudaNarrowphase::contacts()
{ return m_contact[1]->bufferOnDevice(); }

void * CudaNarrowphase::contactPairs()
{ return m_contactPairs->bufferOnDevice();}

void CudaNarrowphase::getContactPairs(BaseBuffer * dst)
{ m_contactPairs->deviceToHost(dst->data(), dst->bufferSize());}

void CudaNarrowphase::getScanResult(BaseBuffer * dst)
{ m_scanValidContacts[0]->deviceToHost(dst->data(), dst->bufferSize());}

CudaNarrowphase::CombinedObjectBuffer * CudaNarrowphase::objectBuffer()
{ return &m_objectBuf; }

CUDABuffer * CudaNarrowphase::contactPairsBuffer()
{ return m_contactPairs; }

CUDABuffer * CudaNarrowphase::contactBuffer()
{ return m_contact[1]; }

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
		m_numElements += m_objects[i]->numTetrahedrons();
		m_numPoints += m_objects[i]->numPoints();
		if(i<m_numObjects-1) {
		    m_objectPointStart[i+1] = m_numPoints;
			m_objectIndexStart[i+1] = m_numElements;
		}
	}
	
	m_objectBuf.m_pos->create(m_numPoints * 12);
	m_objectBuf.m_pos0->create(m_numPoints * 12);
	m_objectBuf.m_vel->create(m_numPoints * 12);
	m_objectBuf.m_mass->create(m_numPoints * 4);
	m_objectBuf.m_ind->create(m_numElements * 16); // 4 ints
	
	m_objectBuf.m_pointCacheLoc->create(CUDANARROWPHASE_MAX_NUMOBJECTS * 4);
	m_objectBuf.m_indexCacheLoc->create(CUDANARROWPHASE_MAX_NUMOBJECTS * 4);
	
	m_objectBuf.m_pointCacheLoc->hostToDevice(&m_objectPointStart[0]);
	m_objectBuf.m_indexCacheLoc->hostToDevice(&m_objectIndexStart[0]);
	
	for(i = 0; i<m_numObjects; i++) {
		CudaTetrahedronSystem * curObj = m_objects[i];
		
		curObj->setDeviceXPtr(m_objectBuf.m_pos, m_objectPointStart[i] * 12);
		curObj->setDeviceXiPtr(m_objectBuf.m_pos0, m_objectPointStart[i] * 12);
		curObj->setDeviceVPtr(m_objectBuf.m_vel, m_objectPointStart[i] * 12);
		curObj->setDeviceMassPtr(m_objectBuf.m_mass, m_objectPointStart[i] * 4);
		curObj->setDeviceTretradhedronIndicesPtr(m_objectBuf.m_ind, m_objectIndexStart[i] * 16);
		
		m_objectBuf.m_pos->hostToDevice(curObj->hostX(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
		m_objectBuf.m_pos0->hostToDevice(curObj->hostXi(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
		m_objectBuf.m_vel->hostToDevice(curObj->hostV(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
		m_objectBuf.m_mass->hostToDevice(curObj->hostMass(), m_objectPointStart[i] * 4, curObj->numPoints() * 4);
		m_objectBuf.m_ind->hostToDevice(curObj->hostTretradhedronIndices(), m_objectIndexStart[i] * 16, curObj->numTetrahedrons() * 16);
	}
}

void CudaNarrowphase::computeContacts(CUDABuffer * overlappingPairBuf, unsigned numOverlappingPairs)
{
    if(numOverlappingPairs < 1) return;
	m_numPairs = numOverlappingPairs;
	
	m_coord->create(nextPow2(numOverlappingPairs * 16));
	m_contact[0]->create(nextPow2(numOverlappingPairs * 48));
	m_contact[1]->create(nextPow2(numOverlappingPairs * 48));
	
	void * overlappingPairs = overlappingPairBuf->bufferOnDevice();
	computeTimeOfImpact(overlappingPairs, numOverlappingPairs);
	squeezeContacts(overlappingPairs, numOverlappingPairs);
}

void CudaNarrowphase::computeTimeOfImpact(void * overlappingPairs, unsigned numOverlappingPairs)
{
	void * dstCoord = m_coord->bufferOnDevice();
	void * dstContact = m_contact[0]->bufferOnDevice();
	void * pos = m_objectBuf.m_pos->bufferOnDevice();
	void * vel = m_objectBuf.m_vel->bufferOnDevice();
	void * ind = m_objectBuf.m_ind->bufferOnDevice();
	narrowphase_computeInitialSeparation((ContactData *)dstContact,
		(uint2 *)overlappingPairs,
		(float3 *)pos,
		(float3 *)vel,
		(uint4 *)ind,
		(uint *)m_objectBuf.m_pointCacheLoc->bufferOnDevice(),
		(uint *)m_objectBuf.m_indexCacheLoc->bufferOnDevice(),
		numOverlappingPairs);
	
	int i;
	for(i=0; i<7; i++) {
	    narrowphase_advanceTimeOfImpactIterative((ContactData *)dstContact,
		(uint2 *)overlappingPairs,
		(float3 *)pos,
		(float3 *)vel,
		(uint4 *)ind,
		(uint *)m_objectBuf.m_pointCacheLoc->bufferOnDevice(),
		(uint *)m_objectBuf.m_indexCacheLoc->bufferOnDevice(),
		numOverlappingPairs);
	}
}

void CudaNarrowphase::squeezeContacts(void * overlappingPairs, unsigned numOverlappingPairs)
{
	void * srcContact = m_contact[0]->bufferOnDevice();
	
	const unsigned scanValidPairLength = iDivUp(numOverlappingPairs, 1024) * 1024;
	
	m_validCounts->create(scanValidPairLength * 4);
	
	void * counts = m_validCounts->bufferOnDevice();
	
	narrowphaseComputeValidPairs((uint *)counts, (ContactData *)srcContact, numOverlappingPairs, scanValidPairLength);
	
	m_scanValidContacts[0]->create(scanValidPairLength * 4);
	m_scanValidContacts[1]->create(scanValidPairLength * 4);
	
	void * scanResult = m_scanValidContacts[0]->bufferOnDevice();
	void * scanIntermediate = m_scanValidContacts[1]->bufferOnDevice();
	scanExclusive((uint *)scanResult, (uint *)counts, (uint *)scanIntermediate, scanValidPairLength / 1024, 1024);

	m_numContacts = ScanUtil::getScanResult(m_validCounts, m_scanValidContacts[0], scanValidPairLength);
	
	if(m_numContacts < 1) return;
	
	m_contactPairs->create(numOverlappingPairs * 8);
	void * dstPairs = m_contactPairs->bufferOnDevice();
	
	void * dstContact = m_contact[1]->bufferOnDevice();
	
	narrowphaseSqueezeContactPairs((uint2 *)dstPairs, (uint2 *)overlappingPairs, 
									(ContactData *)dstContact, (ContactData *)srcContact,
									(uint *)counts, (uint *)scanResult, 
									numOverlappingPairs);
}

void CudaNarrowphase::resetToInitial()
{
    if(m_numPoints < 1) return;
    
    void * dst = m_objectBuf.m_pos->bufferOnDevice();
	void * src = m_objectBuf.m_pos0->bufferOnDevice();
	narrowphaseResetX((float3 *)dst, (float3 *)src, m_numPoints);
}

