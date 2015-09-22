/*
 *  CudaNarrowphase.cpp
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "CudaNarrowphase.h"
#include <CudaMassSystem.h>
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include "narrowphase_implement.h"
#include <CudaScan.h>
#include <DynGlobal.h>
#include <CudaDbgLog.h>
#include <CudaBase.h>
#include <masssystem_impl.h>
#define MaxTOINumIterations 4

CudaNarrowphase::CudaNarrowphase() 
{
	m_numPoints = 0;
    m_numObjects = 0;
	m_numPairs = 0;
	m_numContacts = 0;
    m_objectBuf.m_pos = new CUDABuffer;
    m_objectBuf.m_pos0 = new CUDABuffer;
    m_objectBuf.m_prePos = new CUDABuffer;
    m_objectBuf.m_vel = new CUDABuffer;
    m_objectBuf.m_anchoredVel = new CUDABuffer;
    m_objectBuf.m_mass = new CUDABuffer;
	m_objectBuf.m_anchor = new CUDABuffer;
    m_objectBuf.m_linearImpulse = new CUDABuffer;
    m_objectBuf.m_ind = new CUDABuffer;
	m_objectBuf.m_pointCacheLoc = new CUDABuffer;
	m_objectBuf.m_indexCacheLoc = new CUDABuffer;
	m_contact[0] = new CUDABuffer;
	m_contact[1] = new CUDABuffer;
	m_contactPairs[0] = new CUDABuffer;
	m_contactPairs[1] = new CUDABuffer;
	m_validCounts = new CUDABuffer;
	m_scanValidContacts = new CUDABuffer;
	m_scanIntermediate = new CudaScan;
	m_tetVertPos[0] = new CUDABuffer;
	m_tetVertPos[1] = new CUDABuffer;
	m_tetVertVel[0] = new CUDABuffer;
	m_tetVertVel[1] = new CUDABuffer;
	m_tetVertPrePos = new CUDABuffer;
}

CudaNarrowphase::~CudaNarrowphase() 
{
	delete m_objectBuf.m_pos;
    delete m_objectBuf.m_pos0;
    delete m_objectBuf.m_prePos;
    delete m_objectBuf.m_vel;
    delete m_objectBuf.m_anchoredVel;
    delete m_objectBuf.m_mass;
	delete m_objectBuf.m_anchor;
    delete m_objectBuf.m_linearImpulse;
    delete m_objectBuf.m_ind;
	delete m_objectBuf.m_pointCacheLoc;
	delete m_objectBuf.m_indexCacheLoc;
	delete m_contact[0];
	delete m_contact[1];
	delete m_contactPairs[0];
	delete m_contactPairs[1];
	delete m_validCounts;
	delete m_scanValidContacts;
	delete m_scanIntermediate;
	delete m_tetVertPos[0];
	delete m_tetVertPos[1];
	delete m_tetVertVel[0];
	delete m_tetVertVel[1];
	delete m_tetVertPrePos;
}

const unsigned CudaNarrowphase::numPoints() const
{ return m_numPoints; }

const unsigned CudaNarrowphase::numElements() const
{ return m_numElements; }

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
{ return m_contactPairs[1]->bufferOnDevice();}

void CudaNarrowphase::getContactPairs(BaseBuffer * dst)
{ m_contactPairs[1]->deviceToHost(dst->data(), dst->bufferSize());}

void CudaNarrowphase::getScanResult(BaseBuffer * dst)
{ m_scanValidContacts->deviceToHost(dst->data(), dst->bufferSize());}

CudaNarrowphase::CombinedObjectBuffer * CudaNarrowphase::objectBuffer()
{ return &m_objectBuf; }

CUDABuffer * CudaNarrowphase::contactPairsBuffer()
{ return m_contactPairs[1]; }

CUDABuffer * CudaNarrowphase::contactBuffer()
{ return m_contact[1]; }

void CudaNarrowphase::addMassSystem(CudaMassSystem * tetra)
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
		m_numElements += m_objects[i]->numElements();
		m_numPoints += m_objects[i]->numPoints();
		if(i<m_numObjects-1) {
		    m_objectPointStart[i+1] = m_numPoints;
			m_objectIndexStart[i+1] = m_numElements;
		}
	}
	
	m_objectBuf.m_pos->create(m_numPoints * 12);
	m_objectBuf.m_pos0->create(m_numPoints * 12);
	m_objectBuf.m_prePos->create(m_numPoints * 12);
	m_objectBuf.m_vel->create(m_numPoints * 12);
    m_objectBuf.m_anchoredVel->create(m_numPoints * 12);
	m_objectBuf.m_mass->create(m_numPoints * 4);
	m_objectBuf.m_anchor->create(m_numPoints * 4);
    m_objectBuf.m_linearImpulse->create(m_numPoints * 12);
	m_objectBuf.m_ind->create(m_numElements * 16); // 4 ints
	
	m_objectBuf.m_pointCacheLoc->create(CUDANARROWPHASE_MAX_NUMOBJECTS * 4);
	m_objectBuf.m_indexCacheLoc->create(CUDANARROWPHASE_MAX_NUMOBJECTS * 4);
	
	m_objectBuf.m_pointCacheLoc->hostToDevice(&m_objectPointStart[0]);
	m_objectBuf.m_indexCacheLoc->hostToDevice(&m_objectIndexStart[0]);
	
	for(i = 0; i<m_numObjects; i++) {
		CudaMassSystem * curObj = m_objects[i];
		
		curObj->setDeviceXPtr(m_objectBuf.m_pos, m_objectPointStart[i] * 12);
		curObj->setDeviceXiPtr(m_objectBuf.m_pos0, m_objectPointStart[i] * 12);
		curObj->setDeviceVPtr(m_objectBuf.m_vel, m_objectPointStart[i] * 12);
		curObj->setDeviceVaPtr(m_objectBuf.m_anchoredVel, m_objectPointStart[i] * 12);
        curObj->setDeviceMassPtr(m_objectBuf.m_mass, m_objectPointStart[i] * 4);
		curObj->setDeviceAnchorPtr(m_objectBuf.m_anchor, m_objectPointStart[i] * 4);
        curObj->setDeviceImpulsePtr(m_objectBuf.m_linearImpulse, m_objectPointStart[i] * 4);
		curObj->setDeviceTretradhedronIndicesPtr(m_objectBuf.m_ind, m_objectIndexStart[i] * 16);
		
		m_objectBuf.m_pos->hostToDevice(curObj->hostX(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
		m_objectBuf.m_pos0->hostToDevice(curObj->hostXi(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
        m_objectBuf.m_prePos->hostToDevice(curObj->hostX(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
		m_objectBuf.m_vel->hostToDevice(curObj->hostV(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
		m_objectBuf.m_mass->hostToDevice(curObj->hostMass(), m_objectPointStart[i] * 4, curObj->numPoints() * 4);
		m_objectBuf.m_anchor->hostToDevice(curObj->hostAnchor(), m_objectPointStart[i] * 4, curObj->numPoints() * 4);
		m_objectBuf.m_ind->hostToDevice(curObj->hostTetrahedronIndices(), m_objectIndexStart[i] * 16, curObj->numElements() * 16);
	}
 
    reset();
}

void CudaNarrowphase::computeContacts(CUDABuffer * overlappingPairBuf, 
                                      unsigned numOverlappingPairs,
                                      bool toHandleShallowPenetrating)
{
#if DISABLE_COLLISION_RESOLUTION
	return;
#endif
    m_numContacts = 0;
    m_numPairs = numOverlappingPairs;
    if(numOverlappingPairs < 1) return;
    // std::cout<<" n overlappings "<<numOverlappingPairs<<"\n";
	
	void * overlappingPairs = overlappingPairBuf->bufferOnDevice();
	resetContacts(overlappingPairs, numOverlappingPairs);
	
	computeInitialSeparation();
	
	// std::cout<<" n contact after initial separation "<<
	// m_numContacts = countNoPenetratingContacts(m_numPairs);
    
	computeTimeOfImpact();
	
	//unsigned numPen = countPenetratingContacts(m_numPairs);
	//std::cout<<"  n pens "<<numPen;
	
	if(toHandleShallowPenetrating) handleShallowPenetrations();
	
	m_numContacts = countNoPenetratingContacts(m_numPairs);
		
	if(m_numContacts > 0) {
		// std::cout<<" final squeez contact pairs to "<<m_numContacts<<"\n";
		squeezeContacts(m_numPairs);
	}
}

void CudaNarrowphase::resetContacts(void * overlappingPairs, unsigned numOverlappingPairs)
{
	m_numContacts = 0;
	
	m_contact[0]->create(numOverlappingPairs * 48);
	m_contact[1]->create(numOverlappingPairs * 48);
	m_contactPairs[0]->create(numOverlappingPairs * 8);
	m_contactPairs[1]->create(numOverlappingPairs * 8);
	m_tetVertPos[0]->create(numOverlappingPairs * 2 * 4 * 12);
	m_tetVertPos[1]->create(numOverlappingPairs * 2 * 4 * 12);
	m_tetVertVel[0]->create(numOverlappingPairs * 2 * 4 * 12);
	m_tetVertVel[1]->create(numOverlappingPairs * 2 * 4 * 12);
	m_tetVertPrePos->create(numOverlappingPairs * 2 * 4 * 12);
	
	const unsigned scanValidPairLength = CudaScan::getScanBufferLength(numOverlappingPairs);
	m_scanIntermediate->create(scanValidPairLength);
	m_validCounts->create(scanValidPairLength * 4);
	m_scanValidContacts->create(scanValidPairLength * 4);

	void * pos = m_objectBuf.m_pos->bufferOnDevice();
	void * vel = m_objectBuf.m_vel->bufferOnDevice();
	void * deltaVel = m_objectBuf.m_linearImpulse->bufferOnDevice();
	void * prePos = m_objectBuf.m_prePos->bufferOnDevice();
	void * ind = m_objectBuf.m_ind->bufferOnDevice();
	void * pairPos = m_tetVertPos[0]->bufferOnDevice();
	void * pairVel = m_tetVertVel[0]->bufferOnDevice();
	void * pairPrePos = m_tetVertPrePos->bufferOnDevice();
	
	narrowphase_writePairPosAndVel((float3 *)pairPos,
		(float3 *)pairVel,
		(float3 *)pairPrePos,
		(uint2 *)overlappingPairs,
		(float3 *)pos,
		(float3 *)vel,
		(float3 *)deltaVel,
		(float3 *)prePos,
		(uint4 *)ind,
		(uint *)m_objectBuf.m_pointCacheLoc->bufferOnDevice(),
		(uint *)m_objectBuf.m_indexCacheLoc->bufferOnDevice(),
		numOverlappingPairs);
    CudaBase::CheckCudaError("narrowphase write pair ppre, p and v");
		
	void * dstPair = m_contactPairs[0]->bufferOnDevice();
	narrowphase_writePairs((uint2 *)dstPair, 
		(uint2 *)overlappingPairs,
		numOverlappingPairs);
    CudaBase::CheckCudaError("narrowphase write pair");
}

void CudaNarrowphase::computeInitialSeparation()
{
	void * dstContact = m_contact[0]->bufferOnDevice();
	void * pairPos = m_tetVertPos[0]->bufferOnDevice();
	
	narrowphase_computeInitialSeparation((ContactData *)dstContact,
		(float3 *)pairPos,
		m_numPairs);
    CudaBase::CheckCudaError("narrowphase initial sep");
}

void CudaNarrowphase::computeTimeOfImpact()
{
	void * counts = m_validCounts->bufferOnDevice();
	
	int i;
	for(i=0; i<MaxTOINumIterations; i++) {
		void * dstContact = m_contact[0]->bufferOnDevice();
		void * pairPos = m_tetVertPos[0]->bufferOnDevice();
		void * pairVel = m_tetVertVel[0]->bufferOnDevice();
	
	    narrowphase_advanceTimeOfImpactIterative((ContactData *)dstContact,
		(float3 *)pairPos,
		(float3 *)pairVel,
		m_numPairs);
        CudaBase::CheckCudaError("narrowphase time of impact");
	}
}

void CudaNarrowphase::handleShallowPenetrations()
{
/*  assuming pairs are not penetrating on previous frame
 *  try to find the separation axis
 */
    void * dstContact = m_contact[0]->bufferOnDevice();
	void * pairPos = m_tetVertPrePos->bufferOnDevice();
	
	narrowphase_separateShallowPenetration((ContactData *)dstContact,
		(float3 *)pairPos,
		m_numPairs);
    CudaBase::CheckCudaError("narrowphase initial sep");
}

unsigned CudaNarrowphase::countNoPenetratingContacts(unsigned n)
{
	const unsigned scanValidPairLength = CudaScan::getScanBufferLength(n);
	
	void * counts = m_validCounts->bufferOnDevice();

	narrowphaseComputeValidPairs((uint *)counts, (ContactData *)m_contact[0]->bufferOnDevice(), 
										n, scanValidPairLength);
    CudaBase::CheckCudaError("narrowphase count valid pairs");
	
	return m_scanIntermediate->prefixSum(m_scanValidContacts, 
												m_validCounts, scanValidPairLength);
}

unsigned CudaNarrowphase::countPenetratingContacts(unsigned n)
{
    const unsigned scanValidPairLength = CudaScan::getScanBufferLength(n);
	
	void * counts = m_validCounts->bufferOnDevice();

	narrowphase_computePenetratingPairs((uint *)counts, (ContactData *)m_contact[0]->bufferOnDevice(), 
										n, scanValidPairLength);
    CudaBase::CheckCudaError("narrowphase count penetrating pairs");
	
	return m_scanIntermediate->prefixSum(m_scanValidContacts, 
												m_validCounts, scanValidPairLength);
}

void CudaNarrowphase::squeezeContacts(unsigned numPairs)
{
	void * srcContact = m_contact[0]->bufferOnDevice();
	void * dstContact = m_contact[1]->bufferOnDevice();
	
	void * srcPairs = m_contactPairs[0]->bufferOnDevice();
	void * dstPairs = m_contactPairs[1]->bufferOnDevice();
	
	void * srcPos = m_tetVertPos[0]->bufferOnDevice();
	void * dstPos = m_tetVertPos[1]->bufferOnDevice();
	
	void * srcVel = m_tetVertVel[0]->bufferOnDevice();
	void * dstVel = m_tetVertVel[1]->bufferOnDevice();
	
	void * counts = m_validCounts->bufferOnDevice();
	void * scanResult = m_scanValidContacts->bufferOnDevice();
									
	narrowphase_squeezeContactPosAndVel((float3 *)dstPos, (float3 *)srcPos, 
									(float3 *)dstVel, (float3 *)srcVel,
									(uint2 *)dstPairs, (uint2 *)srcPairs, 
									(ContactData *)dstContact, (ContactData *)srcContact,
									(uint *)counts, (uint *)scanResult, 
									numPairs);
    CudaBase::CheckCudaError("narrowphase squeeze contact");
}

void CudaNarrowphase::reset()
{
    m_numContacts = 0;
    if(numPoints() < 1) return;
    
    void * dst = m_objectBuf.m_pos->bufferOnDevice();
	void * src = m_objectBuf.m_pos0->bufferOnDevice();
    void * vel = m_objectBuf.m_vel->bufferOnDevice();
    void * vel0 = m_objectBuf.m_anchoredVel->bufferOnDevice();
	narrowphaseResetXV((float3 *)dst, (float3 *)src, 
                      (float3 *)vel, (float3 *)vel0, m_numPoints);
}

void CudaNarrowphase::setAnchoredVelocity(Vector3F * src)
{ m_objectBuf.m_anchoredVel->hostToDevice(src, m_numPoints * 12); }

void CudaNarrowphase::upatePosition(float dt)
{
	if(numPoints() < 1) return;
	void * pos = m_objectBuf.m_pos->bufferOnDevice();
	void * vel = m_objectBuf.m_vel->bufferOnDevice();
    void * vela = m_objectBuf.m_anchoredVel->bufferOnDevice();
	void * pre = m_objectBuf.m_prePos->bufferOnDevice();
	void * anchors = m_objectBuf.m_anchor->bufferOnDevice();
	masssystem::integrate((float3 *)pos, 
	                        (float3 *)pre, 
                           (float3 *)vel, 
                           (float3 *)vela, 
						   (uint *)anchors,
                           dt,
                           numPoints());
    CudaBase::CheckCudaError("narrowphase integrate");
}

unsigned CudaNarrowphase::numActiveNodes() const
{ return numPoints() - m_objects[m_numObjects - 1]->numPoints(); }
//:~