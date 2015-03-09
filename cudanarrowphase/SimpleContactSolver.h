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
						void * contacts,
						CUDABuffer * pairBuf,
						void * objectData);
						
						
	CUDABuffer * contactPairHashBuf();
	CUDABuffer * bodySplitLocBuf();
private:
	CUDABuffer * m_sortedInd[2];
	CUDABuffer * m_splitPair;
};
#endif        //  #ifndef SIMPLECONTACTSOLVER_H
