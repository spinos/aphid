/*
 *  CudaNTree.cpp
 *  julia
 *
 *  Created by jian zhang on 3/12/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "CudaNTree.h"
#include <CUDABuffer.h>

namespace aphid {

CudaNTree::CudaNTree(const std::string & name) :
HNTree<cvx::Cube, KdNode4 >(name) 
{
	m_branchPool.reset(new CUDABuffer);
	m_leafPool.reset(new CUDABuffer);
	m_primIndirecion.reset(new CUDABuffer);
	m_ropes.reset(new CUDABuffer);
}

CudaNTree::~CudaNTree() 
{}
	
}
