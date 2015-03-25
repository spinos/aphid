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
#include "scan_implement.h"
#include <ScanUtil.h>
SimpleContactSolver::SimpleContactSolver() 
{
	m_sortedInd[0] = new CUDABuffer;
	m_sortedInd[1] = new CUDABuffer;
	m_splitPair = new CUDABuffer;
	m_bodyCount = new CUDABuffer;
	m_splitInverseMass = new CUDABuffer;
	m_massTensor = new CUDABuffer;
	m_lambda = new CUDABuffer;
	m_projectedLinearVelocity = new CUDABuffer;
	m_projectedAngularVelocity = new CUDABuffer; 
	m_deltaLinearVelocity = new CUDABuffer;
	m_deltaAngularVelocity = new CUDABuffer;
	m_deltaJ = new CUDABuffer;
	m_pntTetHash[0] = new CUDABuffer;
	m_pntTetHash[1] = new CUDABuffer;
}

SimpleContactSolver::~SimpleContactSolver() {}

CUDABuffer * SimpleContactSolver::contactPairHashBuf()
{ return m_sortedInd[0]; }

CUDABuffer * SimpleContactSolver::bodySplitLocBuf()
{ return m_splitPair; }

CUDABuffer * SimpleContactSolver::projectedLinearVelocityBuf()
{ return m_projectedLinearVelocity; }

CUDABuffer * SimpleContactSolver::projectedAngularVelocityBuf()
{ return m_projectedAngularVelocity; }

CUDABuffer * SimpleContactSolver::impulseBuf()
{ return m_lambda; }

CUDABuffer * SimpleContactSolver::deltaLinearVelocityBuf()
{ return m_deltaLinearVelocity; }

CUDABuffer * SimpleContactSolver::deltaAngularVelocityBuf()
{ return m_deltaAngularVelocity; }

CUDABuffer * SimpleContactSolver::deltaJBuf()
{ return m_deltaJ; }

CUDABuffer * SimpleContactSolver::MinvBuf()
{ return m_massTensor; }

CUDABuffer * SimpleContactSolver::pntTetHashBuf()
{ return m_pntTetHash[0]; }

const unsigned SimpleContactSolver::numIterations() const
{ return JACOBI_NUM_ITERATIONS; }

void SimpleContactSolver::solveContacts(unsigned numContacts,
										CUDABuffer * contactBuf,
										CUDABuffer * pairBuf,
										void * objectData)
{
	if(numContacts < 1) return; 
	
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
	
	m_splitInverseMass->create(splitBufLength * 4);
	void * splitMass = m_splitInverseMass->bufferOnDevice();
	
	simpleContactSolverComputeSplitInverseMass((float *)splitMass, 
                            (uint *)bodyCount, 
                            splitBufLength);
	
// one per contact	
	m_lambda->create(numContacts * 4);
	void * lambda = m_lambda->bufferOnDevice();
	
	m_projectedLinearVelocity->create(numContacts * 2 * 12);
	void * projLinVel = m_projectedLinearVelocity->bufferOnDevice();
	
	m_projectedAngularVelocity->create(numContacts * 2 * 12);
	void * projAngVel = m_projectedAngularVelocity->bufferOnDevice();
	
	CudaNarrowphase::CombinedObjectBuffer * objectBuf = (CudaNarrowphase::CombinedObjectBuffer *)objectData;
	void * pos = objectBuf->m_pos->bufferOnDevice();
	void * vel = objectBuf->m_vel->bufferOnDevice();
	void * ind = objectBuf->m_ind->bufferOnDevice();
	void * perObjPointStart = objectBuf->m_pointCacheLoc->bufferOnDevice();
	void * perObjectIndexStart = objectBuf->m_indexCacheLoc->bufferOnDevice();

	m_massTensor->create(numContacts * 4);
	void * Minv = m_massTensor->bufferOnDevice();
	
	void * contacts = contactBuf->bufferOnDevice();
	
// compute projected linear and angular velocity	
// set inital impulse to zero
// calculate Minv
	simpleContactSolverSetContactConstraint((float3 *)projLinVel,
	                                        (float3 *)projAngVel,
	    (float *)lambda, 
	    (float *)Minv,
	    (uint2 *)splits,
	    (uint2 *)pairs,
	    (float3 *)pos,
	    (float3 *)vel,
        (uint4 *)ind,
        (uint * )perObjPointStart,
        (uint * )perObjectIndexStart,
        (float *)splitMass,
	    (ContactData *)contacts,
        numContacts);
	
	m_deltaLinearVelocity->create(splitBufLength * 12);
	m_deltaAngularVelocity->create(splitBufLength * 12);
	
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
	
	m_deltaJ->create(numContacts * JACOBI_NUM_ITERATIONS * 4);
	void * dJ = m_deltaJ->bufferOnDevice();
	
	int i;
	for(i=0; i<JACOBI_NUM_ITERATIONS; i++) {
// compute impulse and velocity changes per contact
	    simpleContactSolverSolveContact((float *)lambda,
	                    (float3 *)deltaLinVel,
	                    (float3 *)deltaAngVel,
	                    (float3 *)projLinVel,
	                    (float3 *)projAngVel,
	                    (uint2 *)splits,
	                    (float *)splitMass,
	                    (float *)Minv,
	                    (ContactData *)contacts,
	                    numContacts,
	                    (float *)dJ,
	                    i);
	    
	    simpleContactSolverAverageVelocities((float3 *)deltaLinVel,
                        (float3 *)deltaAngVel,
                        (uint *)bodyCount,
                        (KeyValuePair *)bodyContactHash, 
                        splitBufLength);
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
	
	void * intermediate = m_pntTetHash[1]->bufferOnDevice();
	RadixSort((KeyValuePair *)pntTetHash, (KeyValuePair *)intermediate, pntHashBufLength, 32);
    	
	simpleContactSolverUpdateVelocity((float3 *)vel,
	                (float3 *)deltaLinVel,
	                (float3 *)deltaAngVel,
	                (KeyValuePair *)pntTetHash,
                    (uint2 *)pairs,
                    (uint2 *)splits,
                    (float3 *)pos,
                    (uint4 *)ind,
                    (uint * )perObjPointStart,
                    (uint * )perObjectIndexStart,
                    numContacts * 2 * 4);
}