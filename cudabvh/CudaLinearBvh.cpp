/*
 *  CudaLinearBvh.cpp
 *  cudabvh
 *
 *  Created by jian zhang on 1/15/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <CudaBase.h>
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include "BvhTriangleMesh.h"
#include "CudaLinearBvh.h"
#include <BvhBuilder.h>
#include <radixsort_implement.h>
#include "bvh_dbg.h"

BvhBuilder * CudaLinearBvh::Builder = 0;

CudaLinearBvh::CudaLinearBvh() 
{ 
	m_numPrimitives = 0;
	m_numActiveInternalNodes = 0;
	m_leafAabbs = new CUDABuffer;
	m_internalNodeAabbs = new CUDABuffer;
	m_leafHash = new CUDABuffer;
	m_leafNodeParentIndices = new CUDABuffer;
	m_internalNodeChildIndices = new CUDABuffer;
	m_internalNodeParentIndices = new CUDABuffer;
	m_rootNodeIndexOnDevice = new CUDABuffer;
	m_distanceInternalNodeFromRoot = new CUDABuffer;
	m_internalNodeNumPrimitives = new CUDABuffer;
	m_maxInternalNodeLevel = 0;

#if DRAW_BVH_HASH
	m_hostLeafHash = new BaseBuffer;
	m_hostLeafBox = new BaseBuffer;
#endif
#if DRAW_BVH_HIERARCHY
	m_hostInternalAabb = new BaseBuffer;
	m_hostInternalChildIndices = new BaseBuffer;
	m_hostInternalDistanceFromRoot = new BaseBuffer;
	m_hostLeafHash = new BaseBuffer;
	m_hostLeafBox = new BaseBuffer;
#endif
}

CudaLinearBvh::~CudaLinearBvh() 
{
	delete m_leafAabbs;
	delete m_internalNodeAabbs;
	delete m_leafHash;
	delete m_leafNodeParentIndices;
	delete m_internalNodeChildIndices;
	delete m_internalNodeParentIndices;
	delete m_rootNodeIndexOnDevice;
	delete m_distanceInternalNodeFromRoot;
	delete m_internalNodeNumPrimitives;
}

void CudaLinearBvh::setNumPrimitives(unsigned n)
{ m_numPrimitives = n; }

void CudaLinearBvh::initOnDevice()
{
    std::cout<<"\n cuda linear bvh init on device";
	m_leafAabbs->create(numLeafNodes() * sizeof(Aabb));
// assume numInternalNodes() >> ReduceMaxBlocks
	m_internalNodeAabbs->create(numInternalNodes() * sizeof(Aabb));
	
	m_leafHash->create(nextPow2(numLeafNodes()) * sizeof(KeyValuePair));
	
	m_leafNodeParentIndices->create(numLeafNodes() * sizeof(int));
	m_internalNodeChildIndices->create(numInternalNodes() * sizeof(int2));
	m_internalNodeParentIndices->create(numInternalNodes() * sizeof(int));
	m_rootNodeIndexOnDevice->create(sizeof(int));
	m_distanceInternalNodeFromRoot->create(numInternalNodes() * sizeof(int));
	m_internalNodeNumPrimitives->create(numInternalNodes() * sizeof(int));

#if DRAW_BVH_HASH
	m_hostLeafBox->create(numPrimitives() * sizeof(Aabb));
	m_hostLeafHash->create(nextPow2(numPrimitives()) * sizeof(KeyValuePair));
#endif

#if DRAW_BVH_HIERARCHY
	m_hostInternalAabb->create(numInternalNodes() * sizeof(Aabb));
	m_hostInternalChildIndices->create(numInternalNodes() * sizeof(int2));
	m_hostInternalDistanceFromRoot->create(numInternalNodes() * sizeof(int));
	m_hostLeafBox->create(numPrimitives() * sizeof(Aabb));
	m_hostLeafHash->create(nextPow2(numPrimitives()) * sizeof(KeyValuePair));
#endif
}

const unsigned CudaLinearBvh::numPrimitives() const
{ return m_numPrimitives; }
	
const unsigned CudaLinearBvh::numLeafNodes() const 
{ return m_numPrimitives; }

const unsigned CudaLinearBvh::numInternalNodes() const 
{ return m_numPrimitives - 1; }

void CudaLinearBvh::getRootNodeIndex(int * dst)
{ m_rootNodeIndexOnDevice->deviceToHost((void *)dst, 4); }

void CudaLinearBvh::getInternalAabbs(BaseBuffer * dst)
{ m_internalNodeAabbs->deviceToHost(dst->data(), m_internalNodeAabbs->bufferSize()); }

void CudaLinearBvh::getInternalDistances(BaseBuffer * dst)
{ m_distanceInternalNodeFromRoot->deviceToHost(dst->data(), m_distanceInternalNodeFromRoot->bufferSize()); }

void CudaLinearBvh::getInternalChildIndex(BaseBuffer * dst)
{ m_internalNodeChildIndices->deviceToHost(dst->data(), m_internalNodeChildIndices->bufferSize()); }

void * CudaLinearBvh::rootNodeIndex()
{ return m_rootNodeIndexOnDevice->bufferOnDevice(); }

void * CudaLinearBvh::internalNodeChildIndices()
{ return m_internalNodeChildIndices->bufferOnDevice(); }

CUDABuffer * CudaLinearBvh::primitiveHashBuf()
{ return m_leafHash; }

void * CudaLinearBvh::internalNodeParentIndices()
{ return m_internalNodeParentIndices->bufferOnDevice(); }

void * CudaLinearBvh::internalNodeAabbs()
{ return m_internalNodeAabbs->bufferOnDevice(); }

void * CudaLinearBvh::leafAabbs()
{ return m_leafAabbs->bufferOnDevice(); }

void * CudaLinearBvh::primitiveAabb()
{ return m_leafAabbs->bufferOnDevice(); }

void * CudaLinearBvh::primitiveHash()
{ return m_leafHash->bufferOnDevice(); }

void CudaLinearBvh::getLeafAabbsAt(char * dst)
{ m_leafAabbs->deviceToHost(dst, 0, m_leafAabbs->bufferSize()); }

void * CudaLinearBvh::leafHash()
{ return m_leafHash->bufferOnDevice(); }

void * CudaLinearBvh::leafNodeParentIndices()
{ return m_leafNodeParentIndices->bufferOnDevice(); }

void * CudaLinearBvh::distanceInternalNodeFromRoot()
{ return m_distanceInternalNodeFromRoot->bufferOnDevice(); }

void * CudaLinearBvh::internalNodeNumPrimitives()
{ return m_internalNodeNumPrimitives->bufferOnDevice(); }

void CudaLinearBvh::update()
{
	Builder->build(this);
}

const unsigned CudaLinearBvh::usedMemory() const
{
	unsigned n = m_leafAabbs->bufferSize();
	n += m_internalNodeAabbs->bufferSize();
	n += m_leafHash->bufferSize();
	n += m_leafNodeParentIndices->bufferSize();
	n += m_internalNodeChildIndices->bufferSize();
	n += m_internalNodeParentIndices->bufferSize();
	n += m_rootNodeIndexOnDevice->bufferSize();
    n += m_distanceInternalNodeFromRoot->bufferSize();
	return n;
}

#if DRAW_BVH_HASH
void * CudaLinearBvh::hostLeafHash()
{ return m_hostLeafHash->data(); }

void * CudaLinearBvh::hostLeafBox()
{ return m_hostLeafBox->data(); }
#endif

void CudaLinearBvh::sendDbgToHost()
{
#if DRAW_BVH_HASH
	m_leafAabbs->deviceToHost(m_hostLeafBox->data(), m_leafAabbs->bufferSize());
	m_leafHash->deviceToHost(m_hostLeafHash->data(), m_leafHash->bufferSize());
#endif

#if DRAW_BVH_HIERARCHY
	m_internalNodeAabbs->deviceToHost(m_hostInternalAabb->data(), m_numActiveInternalNodes * 24);
	m_internalNodeChildIndices->deviceToHost(m_hostInternalChildIndices->data(), m_numActiveInternalNodes * 8);
	m_distanceInternalNodeFromRoot->deviceToHost(m_hostInternalDistanceFromRoot->data(), m_numActiveInternalNodes * 4);
	m_leafAabbs->deviceToHost(m_hostLeafBox->data(), m_leafAabbs->bufferSize());
	m_leafHash->deviceToHost(m_hostLeafHash->data(), m_leafHash->bufferSize());
#endif
}

#if DRAW_BVH_HIERARCHY
void * CudaLinearBvh::hostInternalAabb()
{ return m_hostInternalAabb->data(); }

void * CudaLinearBvh::hostInternalChildIndices()
{ return m_hostInternalChildIndices->data(); }

void * CudaLinearBvh::hostInternalDistanceFromRoot()
{ return m_hostInternalDistanceFromRoot->data(); }

void * CudaLinearBvh::hostPrimitiveHash()
{ return m_hostLeafHash->data(); }

void * CudaLinearBvh::hostPrimitiveAabb()
{ return m_hostLeafBox->data(); }
#endif

void CudaLinearBvh::initRootNode(int * child, float * box)
{
	m_internalNodeChildIndices->hostToDevice(child, 8);
	m_internalNodeAabbs->hostToDevice(box, 24);
    int zero = 0;
    m_internalNodeParentIndices->hostToDevice(&zero, 4);
    m_distanceInternalNodeFromRoot->hostToDevice(&zero, 4);
}

CUDABuffer * CudaLinearBvh::internalChildBuf()
{ return m_internalNodeChildIndices; }

CUDABuffer * CudaLinearBvh::internalAabbBuf()
{ return m_internalNodeAabbs; }

CUDABuffer * CudaLinearBvh::internalParentBuf()
{ return m_internalNodeParentIndices; }

CUDABuffer * CudaLinearBvh::distanceInternalNodeFromRootBuf()
{ return m_distanceInternalNodeFromRoot; }

CUDABuffer * CudaLinearBvh::internalNodeNumPrimitiveBuf()
{ return m_internalNodeNumPrimitives; }

void CudaLinearBvh::setNumActiveInternalNodes(unsigned n)
{ m_numActiveInternalNodes = n; }

const unsigned CudaLinearBvh::numActiveInternalNodes() const
{ return m_numActiveInternalNodes; }

void CudaLinearBvh::setMaxInternalNodeLevel(int n)
{ m_maxInternalNodeLevel = n; }

const int CudaLinearBvh::maxInternalNodeLevel() const
{ return m_maxInternalNodeLevel; }

void CudaLinearBvh::setCostOfTraverse(float x)
{ m_costOfTraverse = x; }

const float CudaLinearBvh::costOfTraverse() const
{ return m_costOfTraverse; }
//:~