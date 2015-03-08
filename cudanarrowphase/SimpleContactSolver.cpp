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
SimpleContactSolver::SimpleContactSolver() {}
SimpleContactSolver::~SimpleContactSolver() {}

void SimpleContactSolver::solveContacts(unsigned numContacts,
										void * contacts,
										CUDABuffer * pairBuf,
										void * objectData)
{
	if(numContacts < 1) return; 
	
	CudaNarrowphase::CombinedObjectBuffer * objectBuf = (CudaNarrowphase::CombinedObjectBuffer *)objectData;
	void * pos = objectBuf->m_pos->bufferOnDevice();
	void * vel = objectBuf->m_vel->bufferOnDevice();
	void * ind = objectBuf->m_ind->bufferOnDevice();
	void * perObjPointStart = objectBuf->m_pointCacheLoc->bufferOnDevice();
	void * perObjectIndexStart = objectBuf->m_indexCacheLoc->bufferOnDevice();
	void * pairs = pairBuf->bufferOnDevice();

	simpleContactSolverStopAtContact((float3 *)vel,
                        (uint2 *)pairs,
						(uint4 *)ind,
						(uint * )perObjPointStart,
                        (uint * )perObjectIndexStart,
                        numContacts);
}