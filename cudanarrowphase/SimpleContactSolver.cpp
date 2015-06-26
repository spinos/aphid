/*
 *  SimpleContactSolver.cpp
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/8/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "SimpleContactSolver.h"
#include "CudaNarrowphase.h"
#include <CUDABuffer.h>
#include "simpleContactSolver_implement.h"
#include <DynGlobal.h>
#include <CudaBase.h>
SimpleContactSolver::SimpleContactSolver() 
{
	m_sortedInd[0] = new CUDABuffer;
	m_sortedInd[1] = new CUDABuffer;
	m_splitPair = new CUDABuffer;
	m_bodyCount = new CUDABuffer;
	m_splitInverseMass = new CUDABuffer;
	m_constraint = new CUDABuffer;
	m_deltaLinearVelocity = new CUDABuffer;
	m_deltaAngularVelocity = new CUDABuffer;
	// m_deltaJ = new CUDABuffer;
	m_relVel = new CUDABuffer;
	m_pntTetHash[0] = new CUDABuffer;
	m_pntTetHash[1] = new CUDABuffer;
	m_numContacts = 0;
	std::cout<<" sizeof struct ContactConstraint "<<sizeof(ContactConstraint)<<"\n";
}

SimpleContactSolver::~SimpleContactSolver() {}

void SimpleContactSolver::initOnDevice()
{
	CudaReduction::initOnDevice();
}

CUDABuffer * SimpleContactSolver::contactPairHashBuf()
{ return m_sortedInd[0]; }

CUDABuffer * SimpleContactSolver::bodySplitLocBuf()
{ return m_splitPair; }

CUDABuffer * SimpleContactSolver::constraintBuf()
{ return m_constraint; }

CUDABuffer * SimpleContactSolver::deltaLinearVelocityBuf()
{ return m_deltaLinearVelocity; }

CUDABuffer * SimpleContactSolver::deltaAngularVelocityBuf()
{ return m_deltaAngularVelocity; }

CUDABuffer * SimpleContactSolver::pntTetHashBuf()
{ return m_pntTetHash[0]; }

CUDABuffer * SimpleContactSolver::splitInverseMassBuf()
{ return m_splitInverseMass; }

const unsigned SimpleContactSolver::numContacts() const
{ return m_numContacts; }

void SimpleContactSolver::solveContacts(unsigned numContacts,
										CUDABuffer * contactBuf,
										CUDABuffer * pairBuf,
										void * objectData)
{
#if DISABLE_COLLISION_RESOLUTION
	return;
#endif
    if(numContacts < 1) return; 
    
	m_numContacts = numContacts;
	const unsigned indBufLength = nextPow2(numContacts * 2);
	
	m_sortedInd[0]->create(indBufLength * 8);	
	m_sortedInd[1]->create(indBufLength * 8);
	
	void * bodyContactHash = m_sortedInd[0]->bufferOnDevice();
	void * pairs = pairBuf->bufferOnDevice();
	
	simpleContactSolverWriteContactIndex((KeyValuePair *)bodyContactHash, (uint *)pairs, numContacts * 2, indBufLength);
	
	void * tmp = m_sortedInd[1]->bufferOnDevice();
	RadixSort((KeyValuePair *)bodyContactHash, (KeyValuePair *)tmp, indBufLength, 32);
	
	m_splitPair->create(numContacts * 8);
	void * splits = m_splitPair->bufferOnDevice();
	
	const unsigned splitBufLength = numContacts * 2;
	simpleContactSolverComputeSplitBufLoc((uint2 *)splits, 
	                        (uint2 *)pairs, 
	                        (KeyValuePair *)bodyContactHash, 
	                        splitBufLength);
	
	m_bodyCount->create(splitBufLength * 4);
	void * bodyCount = m_bodyCount->bufferOnDevice();
	simpleContactSolverCountBody((uint *)bodyCount, 
	                        (KeyValuePair *)bodyContactHash, 
	                        splitBufLength);
							
	int mxcount = 0;
	max<int>(mxcount, (int *)bodyCount, splitBufLength);
    if(mxcount>9) std::cout<<" max count per contact "<<mxcount; 
	int numiterations = mxcount + 2;
	
	m_splitInverseMass->create(splitBufLength * 4);
	void * splitMass = m_splitInverseMass->bufferOnDevice();
	
	CudaNarrowphase::CombinedObjectBuffer * objectBuf = (CudaNarrowphase::CombinedObjectBuffer *)objectData;
	void * pos = objectBuf->m_pos->bufferOnDevice();
	void * vel = objectBuf->m_vel->bufferOnDevice();
	void * mass = objectBuf->m_mass->bufferOnDevice();
	void * ind = objectBuf->m_ind->bufferOnDevice();
	void * perObjPointStart = objectBuf->m_pointCacheLoc->bufferOnDevice();
	void * perObjectIndexStart = objectBuf->m_indexCacheLoc->bufferOnDevice();
	
	simpleContactSolverComputeSplitInverseMass((float *)splitMass,
	                        (uint2 *)splits,
	                        (uint2 *)pairs,
	                        (float *)mass,
	                        (uint4 *)ind,
	                        (uint * )perObjPointStart,
	                        (uint * )perObjectIndexStart,
                            (uint *)bodyCount,
                            splitBufLength);
	
	m_constraint->create(numContacts * 64);
	void * constraint = m_constraint->bufferOnDevice();
	
	void * contacts = contactBuf->bufferOnDevice();
	
	simpleContactSolverSetContactConstraint((ContactConstraint *)constraint,
	    (uint2 *)splits,
	    (uint2 *)pairs,
	    (float3 *)pos,
	    (float3 *)vel,
	    (uint4 *)ind,
        (uint * )perObjPointStart,
        (uint * )perObjectIndexStart,
        (float *)splitMass,
	    (ContactData *)contacts,
        numContacts * 2);
    CudaBase::CheckCudaError("jacobi solver set constraint");
	
	m_deltaLinearVelocity->create(nextPow2(splitBufLength * 12));
	m_deltaAngularVelocity->create(nextPow2(splitBufLength * 12));
	
	void * deltaLinVel = m_deltaLinearVelocity->bufferOnDevice();
	void * deltaAngVel = m_deltaAngularVelocity->bufferOnDevice();
	simpleContactSolverClearDeltaVelocity((float3 *)deltaLinVel, 
	                            (float3 *)deltaAngVel, 
	                            splitBufLength);
	
	/*
	const unsigned scanBufLength = iRound1024(numContacts * 2);
	m_bodyCount->create(scanBufLength * 4);
	m_scanBodyCount[0]->create(scanBufLength * 4);
	m_scanBodyCount[1]->create(scanBufLength * 4);
	
	
	void * scanResult = m_scanBodyCount[0]->bufferOnDevice();
	void * scanIntermediate = m_scanBodyCount[1]->bufferOnDevice();
	scanExclusive((uint *)scanResult, (uint *)bodyCount, (uint *)scanIntermediate, scanBufLength / 1024, 1024);
	
	const unsigned numSplitBodies = ScanUtil::getScanResult(m_bodyCount, m_scanBodyCount[0], scanBufLength);
	*/
	
	int i;
	for(i=0; i< numiterations; i++) {
// compute impulse and velocity changes per contact
        simpleContactSolverSolveContactWoJ((ContactConstraint *)constraint,
	                    (float3 *)deltaLinVel,
	                    (float3 *)deltaAngVel,
	                    (uint2 *)pairs,
	                    (uint2 *)splits,
	                    (float *)splitMass,
	                    (ContactData *)contacts,
	                    (float3 *)pos,
	                    (float3 *)vel,
	                    (uint4 *)ind,
	                    (uint * )perObjPointStart,
	                    (uint * )perObjectIndexStart,
	                    numContacts * 2);
        CudaBase::CheckCudaError("jacobi solver solve impulse");
    
	    simpleContactSolverAverageVelocities((float3 *)deltaLinVel,
                        (float3 *)deltaAngVel,
                        (uint *)bodyCount,
                        (KeyValuePair *)bodyContactHash, 
                        splitBufLength);
        CudaBase::CheckCudaError("jacobi solver average velocity");
	}
	
// 2 tet per contact, 4 pnt per tet, key is pnt index, value is tet index in split
	const unsigned pntHashBufLength = nextPow2(numContacts * 2 * 4);
	m_pntTetHash[0]->create(pntHashBufLength * 8);
	m_pntTetHash[1]->create(pntHashBufLength * 8);
	
	void * pntTetHash = m_pntTetHash[0]->bufferOnDevice();
	
	simpleContactSolverWritePointTetHash((KeyValuePair *)pntTetHash,
	                (uint2 *)pairs,
	                (uint2 *)splits,
	                (uint *)bodyCount,
	                (uint4 *)ind,
	                (uint * )perObjPointStart,
	                (uint * )perObjectIndexStart,
	                numContacts * 2,
	                pntHashBufLength);
	CudaBase::CheckCudaError("jacobi solver point-tetra hash");
    
	void * intermediate = m_pntTetHash[1]->bufferOnDevice();
	RadixSort((KeyValuePair *)pntTetHash, (KeyValuePair *)intermediate, pntHashBufLength, 32);
    	
	simpleContactSolverUpdateVelocity((float3 *)vel,
	                (float3 *)deltaLinVel,
	                (float3 *)deltaAngVel,
	                (KeyValuePair *)pntTetHash,
                    (uint2 *)pairs,
                    (uint2 *)splits,
                    (ContactConstraint *)constraint,
                    (ContactData *)contacts,
                    (float3 *)pos,
                    (uint4 *)ind,
                    (uint * )perObjPointStart,
                    (uint * )perObjectIndexStart,
                    numContacts * 2 * 4);
    CudaBase::CheckCudaError("jacobi solver update velocity");
}