#ifndef CUDALINEARBVH_H
#define CUDALINEARBVH_H

/*
 *  CudaLinearBvh.h
 *  cudabvh
 *
 *  Created by jian zhang on 1/15/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "bvh_common.h"
#include <DynGlobal.h>
// class CudaReduction;
class BaseBuffer;
class CUDABuffer;
class BvhTriangleMesh;
class BvhBuilder;

class CudaLinearBvh {
public:
	CudaLinearBvh();
	virtual ~CudaLinearBvh();
	
	virtual void initOnDevice();
	virtual void update();
	
	const unsigned numPrimitives() const;
	const unsigned numLeafNodes() const;
	const unsigned numInternalNodes() const;
	
	void getRootNodeIndex(int * dst);
	void getLeafAabbsAt(char * dst);
	void getInternalAabbs(BaseBuffer * dst);
	void getInternalDistances(BaseBuffer * dst);
	void getInternalChildIndex(BaseBuffer * dst);
	
	void * rootNodeIndex();
	void * internalNodeChildIndices();
	void * internalNodeParentIndices();
	void * internalNodeAabbs();
	void * internalNodeChildLimit();
	void * leafAabbs();
	void * leafHash();
	void * primitiveAabb();
	void * primitiveHash();
	void * leafNodeParentIndices();
	void * distanceInternalNodeFromRoot();
	void * maxChildElementIndices();
	
	void sendDbgToHost();
	
	const unsigned usedMemory() const;
	
	CUDABuffer * primitiveHashBuf();
	CUDABuffer * internalChildBuf();
	CUDABuffer * internalAabbBuf();
	void setRootChildAndAabb(int * child, float * box);
#if DRAW_BVH_HASH
	void * hostLeafHash();
	void * hostLeafBox();
#endif

#if DRAW_BVH_HIERARCHY
	void * hostInternalAabb();
	void * hostInternalChildIndices();
	void * hostInternalDistanceFromRoot();
	const int hostRootInd() const;
#endif

	static BvhBuilder * Builder;

protected:
	const void setNumPrimitives(unsigned n);
	
private:
	
private:
	CUDABuffer * m_leafAabbs;
	CUDABuffer * m_internalNodeAabbs;
	CUDABuffer * m_leafHash;
	CUDABuffer * m_leafNodeParentIndices;
	CUDABuffer * m_internalNodeChildIndices;
	CUDABuffer * m_internalNodeParentIndices;
	CUDABuffer * m_rootNodeIndexOnDevice;
	CUDABuffer * m_distanceInternalNodeFromRoot;
	CUDABuffer * m_maxChildElementIndices;
	unsigned m_numPrimitives;
#if DRAW_BVH_HASH
	BaseBuffer * m_hostLeafHash;
	BaseBuffer * m_hostLeafBox;
#endif

#if DRAW_BVH_HIERARCHY
	int m_hostRootInd;
	BaseBuffer * m_hostInternalAabb;
	BaseBuffer * m_hostInternalChildIndices;
	BaseBuffer * m_hostInternalDistanceFromRoot;
#endif
};
#endif        //  #ifndef CUDALINEARBVH_H
