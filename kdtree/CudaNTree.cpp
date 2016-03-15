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
#include <CudaBase.h>

namespace aphid {

CudaNTree::CudaNTree(const std::string & name) :
HNTree<cvx::Cube, KdNode4 >(name) 
{
	m_deviceBranch.reset(new CUDABuffer);
	m_deviceLeaf.reset(new CUDABuffer);
	m_deviceIndirection.reset(new CUDABuffer);
	m_deviceRope.reset(new CUDABuffer);
}

CudaNTree::~CudaNTree() 
{}

char CudaNTree::load()
{
	if(!HNTree<cvx::Cube, KdNode4 >::load() ) 
		return 0;
	
	m_deviceBranch->copyFrom<sdb::VectorArray<KdNode4> >(branches() );
	m_deviceLeaf->copyFrom<sdb::VectorArray<knt::TreeLeaf> >(leafNodes() );
	m_deviceIndirection->copyFrom<sdb::VectorArray<int> >(primIndirection() );
	m_deviceRope->copyFrom<sdb::VectorArray<BoundingBox> >(ropes() );
#if 0
	std::cout<<"\n branch buf size "<<m_deviceBranch->bufferSize()
		<<"\n leaf buf size "<<m_deviceLeaf->bufferSize()
		<<"\n prim buf size "<<m_deviceIndirection->bufferSize()
		<<"\n rope buf size "<<m_deviceRope->bufferSize()
		<<"\n cu mem "<<CudaBase::MemoryUsed;
#endif
	return 1;
}
	
void * CudaNTree::deviceBranch() const
{ return m_deviceBranch->bufferOnDevice(); }
	
void * CudaNTree::deviceLeaf() const
{ return m_deviceLeaf->bufferOnDevice(); }

void * CudaNTree::deviceIndirection() const
{ return m_deviceIndirection->bufferOnDevice(); }

void * CudaNTree::deviceRope() const
{ return m_deviceRope->bufferOnDevice(); }
	
}
