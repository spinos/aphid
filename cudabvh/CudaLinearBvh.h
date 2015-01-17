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
class BaseBuffer;
class CUDABuffer;
class BvhTriangleMesh;

class CudaLinearBvh {
public:
	CudaLinearBvh();
	virtual ~CudaLinearBvh();
	
	void setNumLeafNodes(unsigned n);
	void setCombineAabbSecondBlocks(unsigned n);
	void create();
	
	// const unsigned numPoints() const;
	const unsigned numLeafNodes() const;
	const unsigned numInternalNodes() const;
	const Aabb bound() const;
	
	void update();
	
	void getRootNodeIndex(int * dst);
	void getLeafAabbs(BaseBuffer * dst);
	void getInternalAabbs(BaseBuffer * dst);
	void getLeafHash(BaseBuffer * dst);
	void getInternalDistances(BaseBuffer * dst);
	void getInternalChildIndex(BaseBuffer * dst);
	
	void * rootNodeIndex();
	void * internalNodeChildIndices();
	void * internalNodeAabbs();
	void * leafAabbs();
	void * leafHash();
	void * combineAabbsBuffer();
	
	const unsigned usedMemory() const;
	
private:
	void combineAabbSecond();
	void calcLeafHash();
	void buildInternalTree();
	void findMaxDistanceFromRoot();
	void formInternalTreeAabbsIterative();
	
private:
    Aabb m_bound;
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
	CUDABuffer * m_reducedMaxDistance;
	unsigned m_numLeafNodes, m_combineAabbSecondBlocks;
};
#endif        //  #ifndef CUDALINEARBVH_H
