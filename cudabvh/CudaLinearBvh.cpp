/*
 *  CudaLinearBvh.cpp
 *  cudabvh
 *
 *  Created by jian zhang on 1/15/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <CudaBase.h>
#include <CUDABuffer.h>
#include "BvhTriangleMesh.h"
#include "CudaLinearBvh.h"
#include <radixsort_implement.h>
#include "createBvh_implement.h"
#include "reduceBox_implement.h"
#include "reduceRange_implement.h"
#include "bvh_dbg.h"

CudaLinearBvh::CudaLinearBvh() 
{ m_numLeafNodes = 0; }

CudaLinearBvh::~CudaLinearBvh() {}

void CudaLinearBvh::setNumLeafNodes(unsigned n)
{ m_numLeafNodes = n; }

void CudaLinearBvh::setCombineAabbSecondBlocks(unsigned n)
{ m_combineAabbSecondBlocks = n; }
/*
void CudaLinearBvh::setMesh(BvhTriangleMesh * mesh)
{ 
	m_mesh = mesh; 
	init();
}*/

void CudaLinearBvh::create()
{
	m_leafAabbs = new CUDABuffer;
	m_leafAabbs->create(numLeafNodes() * sizeof(Aabb));
	m_internalNodeAabbs = new CUDABuffer;
	// assume numInternalNodes() >> ReduceMaxBlocks
	m_internalNodeAabbs->create(numInternalNodes() * sizeof(Aabb));
	
	m_leafHash[0] = new CUDABuffer;
	m_leafHash[0]->create(numLeafNodes() * sizeof(KeyValuePair));
	m_leafHash[1] = new CUDABuffer;
	m_leafHash[1]->create(numLeafNodes() * sizeof(KeyValuePair));
	
	m_internalNodeCommonPrefixValues = new CUDABuffer;
	m_internalNodeCommonPrefixValues->create(numInternalNodes() * sizeof(uint64));
	m_internalNodeCommonPrefixLengths = new CUDABuffer;
	m_internalNodeCommonPrefixLengths->create(numInternalNodes() * sizeof(int));
	
	m_leafNodeParentIndices = new CUDABuffer;
	m_leafNodeParentIndices->create(numLeafNodes() * sizeof(int));
	m_internalNodeChildIndices = new CUDABuffer;
	m_internalNodeChildIndices->create(numInternalNodes() * sizeof(int2));
	m_internalNodeParentIndices = new CUDABuffer;
	m_internalNodeParentIndices->create(numInternalNodes() * sizeof(int));
	m_rootNodeIndexOnDevice = new CUDABuffer;
	m_rootNodeIndexOnDevice->create(sizeof(int));
	m_distanceInternalNodeFromRoot = new CUDABuffer;
	m_distanceInternalNodeFromRoot->create(numInternalNodes() * sizeof(int));
	
	m_reducedMaxDistance = new CUDABuffer;
	m_reducedMaxDistance->create(ReduceMaxBlocks * sizeof(int));
}

//const unsigned CudaLinearBvh::numPoints() const 
//{ return m_mesh->numVertices(); }

const unsigned CudaLinearBvh::numLeafNodes() const 
{ return m_numLeafNodes; }

const unsigned CudaLinearBvh::numInternalNodes() const 
{ return numLeafNodes() - 1; }

const Aabb CudaLinearBvh::bound() const
{ return m_bound; }

void CudaLinearBvh::getRootNodeIndex(int * dst)
{ m_rootNodeIndexOnDevice->deviceToHost((void *)dst, m_rootNodeIndexOnDevice->bufferSize()); }

//void CudaLinearBvh::getPoints(BaseBuffer * dst)
//{ m_mesh->getVerticesOnDevice(dst); }

void CudaLinearBvh::getLeafAabbs(BaseBuffer * dst)
{ m_leafAabbs->deviceToHost(dst->data(), m_leafAabbs->bufferSize()); }

void CudaLinearBvh::getInternalAabbs(BaseBuffer * dst)
{ m_internalNodeAabbs->deviceToHost(dst->data(), m_internalNodeAabbs->bufferSize()); }

void CudaLinearBvh::getLeafHash(BaseBuffer * dst)
{ m_leafHash[0]->deviceToHost(dst->data(), m_leafHash[0]->bufferSize()); }

void CudaLinearBvh::getInternalDistances(BaseBuffer * dst)
{ m_distanceInternalNodeFromRoot->deviceToHost(dst->data(), m_distanceInternalNodeFromRoot->bufferSize()); }

void CudaLinearBvh::getInternalChildIndex(BaseBuffer * dst)
{ m_internalNodeChildIndices->deviceToHost(dst->data(), m_internalNodeChildIndices->bufferSize()); }

void * CudaLinearBvh::rootNodeIndex()
{ return m_rootNodeIndexOnDevice->bufferOnDevice(); }

void * CudaLinearBvh::internalNodeChildIndices()
{ return m_internalNodeChildIndices->bufferOnDevice(); }

void * CudaLinearBvh::internalNodeAabbs()
{ return m_internalNodeAabbs->bufferOnDevice(); }

void * CudaLinearBvh::leafAabbs()
{ return m_leafAabbs->bufferOnDevice(); }

void * CudaLinearBvh::leafHash()
{ return m_leafHash[0]->bufferOnDevice(); }

void CudaLinearBvh::update()
{
	combineAabbSecond();
	// formLeafAabbs();
	calcLeafHash();
	buildInternalTree();
}

/*
void CudaLinearBvh::formLeafAabbs()
{
    void * cvs = m_mesh->verticesOnDevice();
    void * tri = m_mesh->triangleIndicesOnDevice();
    void * dst = m_leafAabbs->bufferOnDevice();
    bvhCalculateLeafAabbsTriangle((Aabb *)dst, (float3 *)cvs, (uint3 *)tri, numLeafNodes());
}
*/
void CudaLinearBvh::combineAabbSecond()
{
	// void * psrc = m_mesh->verticesOnDevice();
    // void * pdst = m_leafAabbs->bufferOnDevice();
	// 
	// unsigned n = nextPow2(numPoints());
	// unsigned threads, blocks;
	// getReduceBlockThread(blocks, threads, n);
	// 
	// // std::cout<<"n0 "<<n<<" blocks x threads : "<<blocks<<" x "<<threads<<" sharedmem size "<<threads * sizeof(Aabb)<<"\n";
	// 
	// bvhReduceAabbByPoints((Aabb *)pdst, (float3 *)psrc, n, blocks, threads, numPoints());
	// 
	// n = blocks;
	
	void * pdst = internalNodeAabbs();
	unsigned threads, blocks;
	unsigned n = m_combineAabbSecondBlocks;
	while(n > 1) {
		getReduceBlockThread(blocks, threads, n);
		
		// std::cout<<"n "<<n<<" blocks x threads : "<<blocks<<" x "<<threads<<" sharedmem size "<<threads * sizeof(Aabb)<<"\n";
	
		bvhReduceAabbByAabb((Aabb *)pdst, (Aabb *)pdst, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}
	
	cudaMemcpy(&m_bound, pdst, sizeof(Aabb), cudaMemcpyDeviceToHost);
}


void CudaLinearBvh::calcLeafHash()
{
	void * dst = m_leafHash[0]->bufferOnDevice();
	void * src = m_leafAabbs->bufferOnDevice();
	bvhCalculateLeafHash((KeyValuePair *)dst, (Aabb *)src, numLeafNodes(), m_bound);
	void * tmp = m_leafHash[1]->bufferOnDevice();
	RadixSort((KeyValuePair *)dst, (KeyValuePair *)tmp, numLeafNodes(), 32);
}

void CudaLinearBvh::buildInternalTree()
{
	void * morton = m_leafHash[0]->bufferOnDevice();
	void * commonPrefix = m_internalNodeCommonPrefixValues->bufferOnDevice();
	void * commonPrefixLengths = m_internalNodeCommonPrefixLengths->bufferOnDevice();
	
	bvhComputeAdjacentPairCommonPrefix((KeyValuePair *)morton,
										(uint64 *)commonPrefix,
										(int *)commonPrefixLengths,
										numInternalNodes());
	
	void * leafNodeParentIndex = m_leafNodeParentIndices->bufferOnDevice();
	void * internalNodeChildIndex = m_internalNodeChildIndices->bufferOnDevice();
	
	bvhConnectLeafNodesToInternalTree((int *)commonPrefixLengths, 
								(int *)leafNodeParentIndex,
								(int2 *)internalNodeChildIndex, 
								numLeafNodes());
								
	void * internalNodeParentIndex = m_internalNodeParentIndices->bufferOnDevice();
	void * rootInd = m_rootNodeIndexOnDevice->bufferOnDevice();
	bvhConnectInternalTreeNodes((uint64 *)commonPrefix, (int *)commonPrefixLengths,
								(int2 *)internalNodeChildIndex,
								(int *)internalNodeParentIndex,
								(int *)rootInd,
								numInternalNodes());
	
	void * distanceFromRoot = m_distanceInternalNodeFromRoot->bufferOnDevice();
	bvhFindDistanceFromRoot((int *)rootInd, (int *)internalNodeParentIndex,
							(int *)distanceFromRoot, 
							numInternalNodes());
							
	findMaxDistanceFromRoot();						
	formInternalTreeAabbsIterative();
	
	// printLeafInternalNodeConnection();
	// printInternalNodeConnection();
}

void CudaLinearBvh::findMaxDistanceFromRoot()
{
	void * psrc = m_distanceInternalNodeFromRoot->bufferOnDevice();
    void * pdst = m_reducedMaxDistance->bufferOnDevice();
	
	unsigned n = numInternalNodes();
	unsigned threads, blocks;
	getReduceBlockThread(blocks, threads, n);
	
	bvhReduceFindMax((int *)pdst, (int *)psrc, n, blocks, threads);
	
	n = blocks;
	while(n > 1) {
		getReduceBlockThread(blocks, threads, n);
		
		bvhReduceFindMax((int *)pdst, (int *)pdst, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}
}

void CudaLinearBvh::formInternalTreeAabbsIterative()
{
	int maxDistance = -1;
	m_reducedMaxDistance->deviceToHost(&maxDistance, sizeof(int));
	// qDebug()<<"max level "<<maxDistance;
	if(maxDistance < 0) 
		return;
	
	void * distances = m_distanceInternalNodeFromRoot->bufferOnDevice();
	void * boxes = m_leafHash[0]->bufferOnDevice();
	void * internalNodeChildIndex = m_internalNodeChildIndices->bufferOnDevice();
	void * leafNodeAabbs = m_leafAabbs->bufferOnDevice();
	void * internalNodeAabbs = m_internalNodeAabbs->bufferOnDevice();
	for(int distanceFromRoot = maxDistance; distanceFromRoot >= 0; --distanceFromRoot) {		
		bvhFormInternalNodeAabbsAtDistance((int *)distances, (KeyValuePair *)boxes,
											(int2 *)internalNodeChildIndex,
											(Aabb *)leafNodeAabbs, (Aabb *)internalNodeAabbs,
											maxDistance, distanceFromRoot, 
											numInternalNodes());
	}
}

const unsigned CudaLinearBvh::usedMemory() const
{
	unsigned n = m_leafAabbs->bufferSize();
	n += m_internalNodeAabbs->bufferSize();
	n += m_leafHash[0]->bufferSize() * 2;
	n += m_internalNodeCommonPrefixValues->bufferSize();
	n += m_internalNodeCommonPrefixLengths->bufferSize();
	n += m_leafNodeParentIndices->bufferSize();
	n += m_internalNodeChildIndices->bufferSize();
	n += m_internalNodeParentIndices->bufferSize();
	n += m_rootNodeIndexOnDevice->bufferSize();
    n += m_distanceInternalNodeFromRoot->bufferSize();
	n += m_reducedMaxDistance->bufferSize();
	return n;
}
