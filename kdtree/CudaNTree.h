/*
 *  CudaNTree.h
 *  julia
 *
 *  Created by jian zhang on 3/12/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

namespace aphid {

class CUDABuffer;

class CudaNTree {

	CUDABuffer * m_branchPool;
	CUDABuffer * m_leafPool;
	CUDABuffer * m_primIndirecion;
	CUDABuffer * m_ropes;
	int m_numBranches;
	int m_numLeaves;
	int m_numPrimIndices;
	int m_numRopes;
	
public:
	CudaNTree();
	virtual ~CudaNTree();
	
protected:

private:
};

}
