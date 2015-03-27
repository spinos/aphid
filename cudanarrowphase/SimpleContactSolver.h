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
	// CUDABuffer * projectedLinearVelocityBuf();
	// CUDABuffer * projectedAngularVelocityBuf();
	CUDABuffer * impulseBuf();
	CUDABuffer * deltaLinearVelocityBuf();
	CUDABuffer * deltaAngularVelocityBuf();
	CUDABuffer * deltaJBuf();
	CUDABuffer * MinvBuf();
	CUDABuffer * pntTetHashBuf();
	
	const unsigned numIterations() const;
private:
	CUDABuffer * m_sortedInd[2];
	CUDABuffer * m_splitPair;
	CUDABuffer * m_bodyCount;
	CUDABuffer * m_splitInverseMass;
	CUDABuffer * m_massTensor;
	CUDABuffer * m_contraint;
	// CUDABuffer * m_projectedLinearVelocity;
	// CUDABuffer * m_projectedAngularVelocity;
	CUDABuffer * m_lambda;
	CUDABuffer * m_deltaLinearVelocity;
	CUDABuffer * m_deltaAngularVelocity;
	CUDABuffer * m_deltaJ;
	CUDABuffer * m_pntTetHash[2];
};
#endif        //  #ifndef SIMPLECONTACTSOLVER_H
