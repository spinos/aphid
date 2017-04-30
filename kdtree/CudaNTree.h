/*
 *  CudaNTree.h
 *  julia
 *
 *  Created by jian zhang on 3/12/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <boost/scoped_ptr.hpp>
#include <ConvexShape.h>
#include <KdNTree.h>
#include <CUDABuffer.h>
#include <CudaBase.h>

namespace aphid {

template <typename T, typename Tn>
class CudaNTree {

	boost::scoped_ptr<CUDABuffer> m_deviceBranch;
	boost::scoped_ptr<CUDABuffer> m_deviceLeaf;
	boost::scoped_ptr<CUDABuffer> m_deviceIndirection;
	boost::scoped_ptr<CUDABuffer> m_deviceRope;
	boost::scoped_ptr<CUDABuffer> m_devicePrim;
	
public:
	CudaNTree();
	virtual ~CudaNTree();
	
	bool transfer(KdNTree<T, Tn> * src);

	void * deviceBranch() const;
	void * deviceLeaf() const;
	void * deviceIndirection() const;
	void * deviceRope() const;
	void * devicePrim() const;
	
protected:

private:
};

template <typename T, typename Tn>
CudaNTree<T, Tn>::CudaNTree()
{
	m_deviceBranch.reset(new CUDABuffer);
	m_deviceLeaf.reset(new CUDABuffer);
	m_deviceIndirection.reset(new CUDABuffer);
	m_deviceRope.reset(new CUDABuffer);
	m_devicePrim.reset(new CUDABuffer);
}

template <typename T, typename Tn>
CudaNTree<T, Tn>::~CudaNTree() 
{}

template <typename T, typename Tn>
bool CudaNTree<T, Tn>::transfer(KdNTree<T, Tn> * src)
{	
	m_deviceBranch->copyFrom<sdb::VectorArray<KdNode4> >(src->branches() );
	m_deviceLeaf->copyFrom<sdb::VectorArray<knt::TreeLeaf> >(src->leafNodes() );
	m_deviceIndirection->copyFrom<sdb::VectorArray<int> >(src->primIndirection() );
	m_deviceRope->copyFrom<sdb::VectorArray<BoundingBox> >(src->ropes() );
	m_devicePrim->copyFrom<sdb::VectorArray<T> >( *src->source() );
#if 0
	std::cout<<"\n branch buf size "<<m_deviceBranch->bufferSize()
		<<"\n leaf buf size "<<m_deviceLeaf->bufferSize()
		<<"\n prim buf size "<<m_deviceIndirection->bufferSize()
		<<"\n rope buf size "<<m_deviceRope->bufferSize()
		<<"\n prim buf size "<<m_devicePrim->bufferSize()
		<<"\n cu mem "<<CudaBase::MemoryUsed<<" bytes";
#endif
	return 1;
}
	
template <typename T, typename Tn>
void * CudaNTree<T, Tn>::deviceBranch() const
{ return m_deviceBranch->bufferOnDevice(); }
	
template <typename T, typename Tn>
void * CudaNTree<T, Tn>::deviceLeaf() const
{ return m_deviceLeaf->bufferOnDevice(); }

template <typename T, typename Tn>
void * CudaNTree<T, Tn>::deviceIndirection() const
{ return m_deviceIndirection->bufferOnDevice(); }

template <typename T, typename Tn>
void * CudaNTree<T, Tn>::deviceRope() const
{ return m_deviceRope->bufferOnDevice(); }

template <typename T, typename Tn>
void * CudaNTree<T, Tn>::devicePrim() const
{ return m_devicePrim->bufferOnDevice(); }

}
