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
	m_lambda = new CUDABuffer;
	m_projectedLinearVelocity = new CUDABuffer;
	m_projectedAngularVelocity = new CUDABuffer; 
	m_deltaLinearVelocity = new CUDABuffer;
	m_deltaAngularVelocity = new CUDABuffer;
	m_deltaJ = new CUDABuffer;
	m_relV = new CUDABuffer;
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

CUDABuffer * SimpleContactSolver::deltaJBuf()
{ return m_deltaJ; }

CUDABuffer * SimpleContactSolver::relVBuf()
{ return m_relV; }

void SimpleContactSolver::solveContacts(unsigned numContacts,
										CUDABuffer * contactBuf,
										CUDABuffer * pairBuf,
										void * objectData)
{
	if(numContacts < 1) return; 
	
	const unsigned indBufLength = nextPow2(numContacts * 2);
	
	m_sortedInd[0]->create(indBufLength * 8);
	
	m_sortedInd[1]->create(indBufLength * 8);
	
	void * dstInd = m_sortedInd[0]->bufferOnDevice();
	void * pairs = pairBuf->bufferOnDevice();
	
	simpleContactSolverWriteContactIndex((KeyValuePair *)dstInd, (uint *)pairs, numContacts * 2, indBufLength);
	
	void * tmp = m_sortedInd[1]->bufferOnDevice();
	RadixSort((KeyValuePair *)dstInd, (KeyValuePair *)tmp, indBufLength, 32);
	
	m_splitPair->create(numContacts * 8);
	void * splits = m_splitPair->bufferOnDevice();
	
	const unsigned splitBufLength = numContacts * 2;
	simpleContactSolverComputeSplitBufLoc((uint2 *)splits, (uint2 *)pairs, (KeyValuePair *)dstInd, splitBufLength);
	
	m_bodyCount->create(splitBufLength * 4);
	void * dstCount = m_bodyCount->bufferOnDevice();
	simpleContactSolverCountBody((uint *)dstCount, (KeyValuePair *)dstInd, splitBufLength);
	
	m_splitInverseMass->create(splitBufLength * 4);
	void * splitMass = m_splitInverseMass->bufferOnDevice();
	
	simpleContactSolverComputeSplitInverseMass((float *)splitMass, (uint *)dstCount, splitBufLength);
	
// A and B per contact	
	m_lambda->create(numContacts * 8);
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

// compute projected linear and angular velocity	
// set inital impulse to zero
	simpleContactSolverSetContactConstraint((float3 *)projLinVel,
	                                        (float3 *)projAngVel,
	    (float *)lambda, 
	    (uint2 *)splits,
	    (uint2 *)pairs,
	    (float3 *)pos,
	    (float3 *)vel,
        (uint4 *)ind,
        (uint * )perObjPointStart,
        (uint * )perObjectIndexStart,
        numContacts);
	
	m_deltaLinearVelocity->create(splitBufLength * 12);
	m_deltaAngularVelocity->create(splitBufLength * 12);
	
	void * deltaLinVel = m_deltaLinearVelocity->bufferOnDevice();
	void * deltaAngVel = m_deltaAngularVelocity->bufferOnDevice();
	simpleContactSolverClearDeltaVelocity((float3 *)deltaLinVel, (float3 *)deltaAngVel, splitBufLength);
	
	/*
	const unsigned scanBufLength = iRound1024(numContacts * 2);
	m_bodyCount->create(scanBufLength * 4);
	m_scanBodyCount[0]->create(scanBufLength * 4);
	m_scanBodyCount[1]->create(scanBufLength * 4);
	
	
	void * scanResult = m_scanBodyCount[0]->bufferOnDevice();
	void * scanIntermediate = m_scanBodyCount[1]->bufferOnDevice();
	scanExclusive((uint *)scanResult, (uint *)dstCount, (uint *)scanIntermediate, scanBufLength / 1024, 1024);
	
	const unsigned numSplitBodies = ScanUtil::getScanResult(m_bodyCount, m_scanBodyCount[0], scanBufLength);
	*/
	
	m_deltaJ->create(numContacts * 8);
	void * dJ = m_deltaJ->bufferOnDevice();
	
	m_relV->create(numContacts * JACOBI_NUM_ITERATIONS * 12);
	void * relV = m_relV->bufferOnDevice();
	
	void * contacts = contactBuf->bufferOnDevice();
	int i;
	for(i=0; i<JACOBI_NUM_ITERATIONS; i++) {
// compute impulse and velocity changes per contact
	    simpleContactSolverSolveContact((float *)lambda,
	                    (float3 *)projLinVel,
	                    (float3 *)projAngVel,
	                    (uint2 *)splits,
	                    (float *)splitMass,
	                    (ContactData *)contacts,
	                    (float *)dJ,
	                    numContacts,
	                    (float3 *)relV,
	                    i);
	}
	
	simpleContactSolverStopAtContact((float3 *)vel,
                    (uint2 *)pairs,
                    (uint4 *)ind,
                    (uint * )perObjPointStart,
                    (uint * )perObjectIndexStart,
                    numContacts);
}