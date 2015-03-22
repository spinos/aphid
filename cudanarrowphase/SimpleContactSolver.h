#ifndef SIMPLECONTACTSOLVER_H
#define SIMPLECONTACTSOLVER_H

/*
 *  SimpleContactSolver.h
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/8/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
class CUDABuffer;
class SimpleContactSolver {
public:
	SimpleContactSolver();
	virtual ~SimpleContactSolver();
	void solveContacts(unsigned numContacts,
						CUDABuffer * contactBuf,
						CUDABuffer * pairBuf,
						void * objectData);
						
						
	CUDABuffer * contactPairHashBuf();
	CUDABuffer * bodySplitLocBuf();
	CUDABuffer * contactLinearVelocityBuf();
	CUDABuffer * contactAngularVelocityBuf();
	CUDABuffer * impulseBuf();
	
private:
	CUDABuffer * m_sortedInd[2];
	CUDABuffer * m_splitPair;
	CUDABuffer * m_bodyCount;
	CUDABuffer * m_splitInverseMass;
	CUDABuffer * m_contactLinearVelocity;
	CUDABuffer * m_contactAngularVelocity;
	CUDABuffer * m_lambda;
	CUDABuffer * m_deltaLinearVelocity;
	CUDABuffer * m_deltaAngularVelocity;
};
#endif        //  #ifndef SIMPLECONTACTSOLVER_H
