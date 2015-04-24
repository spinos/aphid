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
class CudaReduction;
class BaseBuffer;
class CUDABuffer;
class BvhTriangleMesh;

class CudaLinearBvh {
public:
	CudaLinearBvh();
	virtual ~CudaLinearBvh();
	
	void setNumLeafNodes(unsigned n);
	virtual void initOnDevice();
	virtual void update();
	
	const unsigned numLeafNodes() const;
	const unsigned numInternalNodes() const;
	
	const Aabb getAabb() const;
	void getRootNodeIndex(int * dst);
	void getLeafAabbsAt(char * dst);
	void getInternalAabbs(BaseBuffer * dst);
	void getInternalDistances(BaseBuffer * dst);
	void getInternalChildIndex(BaseBuffer * dst);
	
	void * rootNodeIndex();
	void * internalNodeChildIndices();
	void * internalNodeAabbs();
	void * leafAabbs();
	void * leafHash();
	
	void sendDbgToHost();
	
	const unsigned usedMemory() const;
	
#if DRAW_BVH_HASH
	void * hostLeafHash();
	void * hostLeafBox();
#endif

#if DRAW_BVH_HIERARCHY
	void * hostInternalAabb();
#endif

protected:
	CudaReduction * reducer();
	float * aabbPtr();
private:
	void computeAndSortLeafHash();
	void buildInternalTree();
	void formInternalTreeAabbsIterative(int maxDistance);
	void computeAabb();
private:
	Aabb m_bounding;
    CUDABuffer * m_leafAabbs;
	CUDABuffer * m_internalNodeAabbs;
	CUDABuffer * m_leafHash[2];
	CUDABuffer * m_internalNodeCommonPrefixValues;
	CUDABuffer * m_internalNodeCommonPrefixLengths;
	CUDABuffer * m_leafNodeParentIndices;
	CUDABuffer * m_internalNodeChildIndices;
	CUDABuffer * m_internalNodeParentIndices;
	CUDABuffer * m_rootNodeIndexOnDevice;
    CUDABuffer * m_distanceInternalNodeFromRoot;
	CudaReduction * m_findMaxDistance;
	unsigned m_numLeafNodes;
#if DRAW_BVH_HASH
	BaseBuffer * m_hostLeafHash;
	BaseBuffer * m_hostLeafBox;
#endif

#if DRAW_BVH_HIERARCHY
	BaseBuffer * m_hostInternalAabb;
	
	
#endif
};
#endif        //  #ifndef CUDALINEARBVH_H
