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
	void initOnDevice();
	void solveContacts(unsigned numContacts,
						CUDABuffer * contactBuf,
						CUDABuffer * pairBuf,
						void * objectData);
						
						
	CUDABuffer * contactPairHashBuf();
	CUDABuffer * bodySplitLocBuf();
	CUDABuffer * constraintBuf();
	CUDABuffer * deltaLinearVelocityBuf();
	CUDABuffer * deltaAngularVelocityBuf();
	CUDABuffer * deltaJBuf();
	CUDABuffer * pntTetHashBuf();
	
	const unsigned numIterations() const;
	const unsigned numContacts() const;
private:
	CUDABuffer * m_sortedInd[2];
	CUDABuffer * m_splitPair;
	CUDABuffer * m_bodyCount;
	CUDABuffer * m_splitInverseMass;
	CUDABuffer * m_constraint;
	CUDABuffer * m_deltaLinearVelocity;
	CUDABuffer * m_deltaAngularVelocity;
	CUDABuffer * m_deltaJ;
	CUDABuffer * m_relVel;
	CUDABuffer * m_pntTetHash[2];
	unsigned m_numContacts;
};
#endif        //  #ifndef SIMPLECONTACTSOLVER_H
