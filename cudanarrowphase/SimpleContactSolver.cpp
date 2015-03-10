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
}

SimpleContactSolver::~SimpleContactSolver() {}

CUDABuffer * SimpleContactSolver::contactPairHashBuf()
{ return m_sortedInd[0]; }

CUDABuffer * SimpleContactSolver::bodySplitLocBuf()
{ return m_splitPair; }

void SimpleContactSolver::solveContacts(unsigned numContacts,
										void * contacts,
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
	
	/*
	const unsigned scanBufLength = iRound1024(numContacts * 2);
	m_bodyCount->create(scanBufLength * 4);
	m_scanBodyCount[0]->create(scanBufLength * 4);
	m_scanBodyCount[1]->create(scanBufLength * 4);
	
	void * dstCount = m_bodyCount->bufferOnDevice();
	simpleContactSolverCountUniqueBody((uint *)dstCount, (KeyValuePair *)dstInd, numContacts * 2, scanBufLength);
	
	void * scanResult = m_scanBodyCount[0]->bufferOnDevice();
	void * scanIntermediate = m_scanBodyCount[1]->bufferOnDevice();
	scanExclusive((uint *)scanResult, (uint *)dstCount, (uint *)scanIntermediate, scanBufLength / 1024, 1024);
	
	const unsigned numSplitBodies = ScanUtil::getScanResult(m_bodyCount, m_scanBodyCount[0], scanBufLength);
	*/
	
	CudaNarrowphase::CombinedObjectBuffer * objectBuf = (CudaNarrowphase::CombinedObjectBuffer *)objectData;
	void * pos = objectBuf->m_pos->bufferOnDevice();
	void * vel = objectBuf->m_vel->bufferOnDevice();
	void * ind = objectBuf->m_ind->bufferOnDevice();
	void * perObjPointStart = objectBuf->m_pointCacheLoc->bufferOnDevice();
	void * perObjectIndexStart = objectBuf->m_indexCacheLoc->bufferOnDevice();
	
	simpleContactSolverStopAtContact((float3 *)vel,
                        (uint2 *)pairs,
						(uint4 *)ind,
						(uint * )perObjPointStart,
                        (uint * )perObjectIndexStart,
                        numContacts);
}