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
CudaDbgLog nplg("nph.txt");

struct SSimplex {
    float3 p[4];
	float3 pA[4];
	float3 pB[4];
	int dimension;
};

struct SContactData {
    float4 separateAxis;
    float3 localA;
    float3 localB;
    float toi;
};

CudaNarrowphase::CudaNarrowphase() 
{
	// std::cout<<" size of simplex "<<sizeof(SSimplex)<<" \n";
	// std::cout<<" size of ctc "<<sizeof(ClosestPointTestContext)<<" \n";
	// std::cout<<" size of contact "<<sizeof(SContactData)<<" \n";
    m_bufferId = 0;
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
}

CudaNarrowphase::~CudaNarrowphase() 
{
	delete m_objectBuf.m_pos;
    delete m_objectBuf.m_pos0;
    delete m_objectBuf.m_vel;
    delete m_objectBuf.m_mass;
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
}

const unsigned CudaNarrowphase::numPoints() const
{ return m_numPoints; }

const unsigned CudaNarrowphase::numElements() const
{ return m_numElements; }

void CudaNarrowphase::getContact0(BaseBuffer * dst)
{ m_contact[otherBufferId()]->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaNarrowphase::getContact(BaseBuffer * dst)
{ m_contact[bufferId()]->deviceToHost(dst->data(), dst->bufferSize()); }

void CudaNarrowphase::getContactCounts(BaseBuffer * dst)
{ m_validCounts->deviceToHost(dst->data(), dst->bufferSize()); }

const unsigned CudaNarrowphase::numPairs() const
{ return m_numPairs; }

const unsigned CudaNarrowphase::numContacts() const
{ return m_numContacts; }

void * CudaNarrowphase::contacts0()
{ return m_contact[otherBufferId()]->bufferOnDevice(); }

void * CudaNarrowphase::contacts()
{ return m_contact[bufferId()]->bufferOnDevice(); }

void * CudaNarrowphase::contactPairs()
{ return m_contactPairs[bufferId()]->bufferOnDevice();}

void CudaNarrowphase::getContactPairs(BaseBuffer * dst)
{ m_contactPairs[bufferId()]->deviceToHost(dst->data(), dst->bufferSize());}

void CudaNarrowphase::getScanResult(BaseBuffer * dst)
{ m_scanValidContacts->deviceToHost(dst->data(), dst->bufferSize());}

CudaNarrowphase::CombinedObjectBuffer * CudaNarrowphase::objectBuffer()
{ return &m_objectBuf; }

CUDABuffer * CudaNarrowphase::contactPairsBuffer()
{ return m_contactPairs[bufferId()]; }

CUDABuffer * CudaNarrowphase::contactBuffer()
{ return m_contact[bufferId()]; }

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
	m_objectBuf.m_vel->create(m_numPoints * 12);
	m_objectBuf.m_mass->create(m_numPoints * 4);
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
		curObj->setDeviceMassPtr(m_objectBuf.m_mass, m_objectPointStart[i] * 4);
		curObj->setDeviceTretradhedronIndicesPtr(m_objectBuf.m_ind, m_objectIndexStart[i] * 16);
		
		m_objectBuf.m_pos->hostToDevice(curObj->hostX(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
		m_objectBuf.m_pos0->hostToDevice(curObj->hostXi(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
		m_objectBuf.m_vel->hostToDevice(curObj->hostV(), m_objectPointStart[i] * 12, curObj->numPoints() * 12);
		m_objectBuf.m_mass->hostToDevice(curObj->hostMass(), m_objectPointStart[i] * 4, curObj->numPoints() * 4);
		m_objectBuf.m_ind->hostToDevice(curObj->hostTetrahedronIndices(), m_objectIndexStart[i] * 16, curObj->numElements() * 16);
	}
	
	const unsigned estimatedN = m_numElements * 2;
	m_contact[0]->create(estimatedN * 48);
	m_contact[1]->create(estimatedN * 48);
	m_contactPairs[0]->create(estimatedN * 8);
	m_contactPairs[1]->create(estimatedN * 8);
	m_tetVertPos[0]->create(estimatedN * 2 * 4 * 12);
	m_tetVertPos[1]->create(estimatedN * 2 * 4 * 12);
	m_tetVertVel[0]->create(estimatedN * 2 * 4 * 12);
	m_tetVertVel[1]->create(estimatedN * 2 * 4 * 12);
}

void CudaNarrowphase::computeContacts(CUDABuffer * overlappingPairBuf, unsigned numOverlappingPairs)
{
#if DISABLE_COLLISION_RESOLUTION
	return;
#endif
    if(numOverlappingPairs < 1) return;
    
    std::cout<<" n overlappings "<<numOverlappingPairs<<"\n";
	
	void * overlappingPairs = overlappingPairBuf->bufferOnDevice();
	resetContacts(overlappingPairs, numOverlappingPairs);
	
	computeInitialSeparation();
	
	// std::cout<<" n contact after initial separation "<<
	m_numContacts = countValidContacts(m_contact[bufferId()], numOverlappingPairs);
	
    if(m_numContacts < 1) {
		return;
	}
    
	computeTimeOfImpact();
	
	if(m_numPairs < 1) {
		m_numContacts = 0;
		return;
	}
	
	m_numContacts = countValidContacts(m_contact[bufferId()], m_numPairs);
	
	// std::cout<<" n contacts "<<m_numContacts<<"\n";
	
	if(m_numContacts < 1) return;
		
	if(m_numContacts < m_numPairs) {
		std::cout<<" final squeez contact pairs to "<<m_numContacts<<"\n";

		squeezeContacts(m_numPairs);
		swapBuffer();
	}
}

void CudaNarrowphase::resetContacts(void * overlappingPairs, unsigned numOverlappingPairs)
{
	m_numPairs = numOverlappingPairs;
	m_numContacts = 0;
	m_bufferId = 0;
	
	m_contact[0]->create(numOverlappingPairs * 48);
	m_contact[1]->create(numOverlappingPairs * 48);
	m_contactPairs[0]->create(numOverlappingPairs * 8);
	m_contactPairs[1]->create(numOverlappingPairs * 8);
	m_tetVertPos[0]->create(numOverlappingPairs * 2 * 4 * 12);
	m_tetVertPos[1]->create(numOverlappingPairs * 2 * 4 * 12);
	m_tetVertVel[0]->create(numOverlappingPairs * 2 * 4 * 12);
	m_tetVertVel[1]->create(numOverlappingPairs * 2 * 4 * 12);
	
	const unsigned scanValidPairLength = CudaScan::getScanBufferLength(numOverlappingPairs);
	m_scanIntermediate->create(scanValidPairLength);
	m_validCounts->create(scanValidPairLength * 4);
	m_scanValidContacts->create(scanValidPairLength * 4);

	void * pos = m_objectBuf.m_pos->bufferOnDevice();
	void * vel = m_objectBuf.m_vel->bufferOnDevice();
	void * ind = m_objectBuf.m_ind->bufferOnDevice();
	void * pairPos = m_tetVertPos[bufferId()]->bufferOnDevice();
	void * pairVel = m_tetVertVel[bufferId()]->bufferOnDevice();
	
	narrowphase_writePairPosAndVel((float3 *)pairPos,
		(float3 *)pairVel,
		(uint2 *)overlappingPairs,
		(float3 *)pos,
		(float3 *)vel,
		(uint4 *)ind,
		(uint *)m_objectBuf.m_pointCacheLoc->bufferOnDevice(),
		(uint *)m_objectBuf.m_indexCacheLoc->bufferOnDevice(),
		numOverlappingPairs);
    CudaBase::CheckCudaError("narrowphase write pair p and v");
		
	void * dstPair = m_contactPairs[bufferId()]->bufferOnDevice();
	narrowphase_writePairs((uint2 *)dstPair, 
		(uint2 *)overlappingPairs,
		numOverlappingPairs);
    CudaBase::CheckCudaError("narrowphase write pair");
}

void CudaNarrowphase::computeInitialSeparation()
{
	void * dstContact = m_contact[bufferId()]->bufferOnDevice();
	void * pairPos = m_tetVertPos[bufferId()]->bufferOnDevice();
	
	narrowphase_computeInitialSeparation((ContactData *)dstContact,
		(float3 *)pairPos,
		m_numPairs);
    CudaBase::CheckCudaError("narrowphase initial sep");
}

void CudaNarrowphase::computeTimeOfImpact()
{
	unsigned lastNumPairs = m_numPairs;
	
	void * counts = m_validCounts->bufferOnDevice();
	
	int i;
	for(i=0; i<DynGlobal::MaxTOINumIterations; i++) {
		void * dstContact = m_contact[bufferId()]->bufferOnDevice();
		void * pairPos = m_tetVertPos[bufferId()]->bufferOnDevice();
		void * pairVel = m_tetVertVel[bufferId()]->bufferOnDevice();
	
	    narrowphase_advanceTimeOfImpactIterative((ContactData *)dstContact,
		(float3 *)pairPos,
		(float3 *)pairVel,
		lastNumPairs);
        CudaBase::CheckCudaError("narrowphase time of impact");
		
		if(i<1) {
			m_numPairs = countValidContacts(m_contact[bufferId()], lastNumPairs);
			std::cout<<" squeez contact pairs "<<lastNumPairs<<" to "<<m_numPairs<<"\n";
			squeezeContacts(lastNumPairs);
			swapBuffer();
			lastNumPairs = m_numPairs;
		}
		
		/*
		if(m_numPairs < lastNumPairs>>2) {
			// std::cout<<" squeez contact pairs "<<lastNumPairs<<" to "<<m_numPairs<<"\n";

			//nplg.writeVec3(m_tetVertPos[bufferId()], lastNumPairs<<3, "posb4", CudaDbgLog::FAlways);
			//nplg.writeVec3(m_tetVertVel[bufferId()], lastNumPairs<<3, "velb4", CudaDbgLog::FAlways);
	
			squeezeContacts(lastNumPairs);
			swapBuffer();
			
			//nplg.writeVec3(m_tetVertPos[bufferId()], m_numPairs<<3, "posaft", CudaDbgLog::FAlways);
			//nplg.writeVec3(m_tetVertVel[bufferId()], m_numPairs<<3, "velaft", CudaDbgLog::FAlways);

			lastNumPairs = m_numPairs;
		}*/
	}
}

unsigned CudaNarrowphase::countValidContacts(CUDABuffer * contactBuf, unsigned n)
{
	const unsigned scanValidPairLength = CudaScan::getScanBufferLength(n);
	
	void * counts = m_validCounts->bufferOnDevice();

	narrowphaseComputeValidPairs((uint *)counts, (ContactData *)contactBuf->bufferOnDevice(), 
										n, scanValidPairLength);
    CudaBase::CheckCudaError("narrowphase count valid");
	
	return m_scanIntermediate->prefixSum(m_scanValidContacts, 
												m_validCounts, scanValidPairLength);
}

void CudaNarrowphase::squeezeContacts(unsigned numPairs)
{
	//nplg.writeUInt(m_validCounts, numPairs, "counts", CudaDbgLog::FAlways);
	//nplg.writeUInt(m_scanValidContacts, numPairs, "scan_result", CudaDbgLog::FAlways);
	
	void * srcContact = m_contact[bufferId()]->bufferOnDevice();
	void * dstContact = m_contact[otherBufferId()]->bufferOnDevice();
	
	void * srcPairs = m_contactPairs[bufferId()]->bufferOnDevice();
	void * dstPairs = m_contactPairs[otherBufferId()]->bufferOnDevice();
	
	void * srcPos = m_tetVertPos[bufferId()]->bufferOnDevice();
	void * dstPos = m_tetVertPos[otherBufferId()]->bufferOnDevice();
	
	void * srcVel = m_tetVertVel[bufferId()]->bufferOnDevice();
	void * dstVel = m_tetVertVel[otherBufferId()]->bufferOnDevice();
	
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

void CudaNarrowphase::resetToInitial()
{
    if(m_numPoints < 1) return;
    
    void * dst = m_objectBuf.m_pos->bufferOnDevice();
	void * src = m_objectBuf.m_pos0->bufferOnDevice();
	narrowphaseResetX((float3 *)dst, (float3 *)src, m_numPoints);
}

void CudaNarrowphase::swapBuffer()
{ m_bufferId = (m_bufferId + 1) & 1; }

const unsigned CudaNarrowphase::bufferId() const
{ return m_bufferId; }
	
const unsigned CudaNarrowphase::otherBufferId() const
{ return (m_bufferId + 1) & 1; }
