/*
 *  BvhSolver.cpp
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtCore>
#include <CudaBase.h>
#include <CUDABuffer.h>
#include "BvhTriangleMesh.h"

#include "BvhSolver.h"
#include "createBvh_implement.h"
#include "traverseBvh_implement.h"
#include "reduceBox_implement.h"
#include "reduceRange_implement.h"

BvhSolver::BvhSolver(QObject *parent) : BaseSolverThread(parent) 
{
	m_alpha = 0;
	m_isValid = 0;
}

BvhSolver::~BvhSolver() {}

void BvhSolver::setMesh(BvhTriangleMesh * mesh)
{ m_mesh = mesh; }

void BvhSolver::createEdges(BaseBuffer * onhost, uint n)
{
	m_numLeafNodes = n;
	m_edgeContactIndices = new CUDABuffer;
	m_edgeContactIndices->create(onhost->bufferSize());
	m_edgeContactIndices->hostToDevice(onhost->data(), onhost->bufferSize());
}

void BvhSolver::createRays(uint m, uint n)
{
	m_numRays = m * n;
	m_rayDim = m;
	m_rays = new CUDABuffer;
	m_rays->create(m_numRays * sizeof(RayInfo));
	
	m_ntests = new CUDABuffer;
	m_ntests->create(m_numRays * sizeof(float));
}

void BvhSolver::init()
{
	qDebug()<<"solverinit";

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
	
	qDebug()<<"num points "<<numPoints();
	qDebug()<<"num internal nodes "<<numInternalNodes();
	qDebug()<<"num leaf nodes "<<numLeafNodes();
}

void BvhSolver::stepPhysics(float dt)
{
	m_mesh->update();
	combineAabb();
	formLeafAabbs();
	calcLeafHash();
	buildInternalTree();
	formRays();
	rayTraverse();
	m_isValid = 1;
}

void BvhSolver::formLeafAabbs()
{
    void * cvs = m_mesh->verticesOnDevice();
    void * edges = m_edgeContactIndices->bufferOnDevice();
    void * dst = m_leafAabbs->bufferOnDevice();
    bvhCalculateLeafAabbs((Aabb *)dst, (float3 *)cvs, (EdgeContact *)edges, numLeafNodes(), numPoints());
}

void BvhSolver::combineAabb()
{
	void * psrc = m_mesh->verticesOnDevice();
    void * pdst = m_internalNodeAabbs->bufferOnDevice();
	
	unsigned n = numLeafNodes();
	unsigned threads, blocks;
	getReduceBlockThread(blocks, threads, n);
	
	// std::cout<<"n0 "<<n<<" blocks X threads : "<<blocks<<" X "<<threads<<"\n";
	
	bvhReduceAabbByPoints((Aabb *)pdst, (float3 *)psrc, n, blocks, threads);
	
	n = blocks;
	while(n > 1) {
		getReduceBlockThread(blocks, threads, n);
		
		// std::cout<<"n "<<n<<" blocks X threads : "<<blocks<<" X "<<threads<<"\n";
	
		bvhReduceAabbByAabb((Aabb *)pdst, (Aabb *)pdst, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}
}

void BvhSolver::calcLeafHash()
{
	void * dst = m_leafHash[0]->bufferOnDevice();
	void * src = m_leafAabbs->bufferOnDevice();
	void * box = m_internalNodeAabbs->bufferOnDevice();
	bvhCalculateLeafHash((KeyValuePair *)dst, (Aabb *)src, numLeafNodes(), (Aabb *)box);
	void * tmp = m_leafHash[1]->bufferOnDevice();
	RadixSort((KeyValuePair *)dst, (KeyValuePair *)tmp, numLeafNodes(), 32);
}

void BvhSolver::buildInternalTree()
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

void BvhSolver::findMaxDistanceFromRoot()
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

void BvhSolver::formInternalTreeAabbsIterative()
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

void BvhSolver::formRays()
{
	void * rays = m_rays->bufferOnDevice();
	
	float3 ori; 
	ori.x = sin(m_alpha * 0.2f) * 60.f;
	ori.y = 60.f + sin(m_alpha * 0.1f) * 20.f;
	ori.z = cos(m_alpha * 0.2f) * 30.f;
	bvhTestRay((RayInfo *)rays, ori, 10.f, m_rayDim, m_numRays);
}

void BvhSolver::rayTraverse()
{
	void * rays = m_rays->bufferOnDevice();
	void * rootNodeIndex = m_rootNodeIndexOnDevice->bufferOnDevice();
	void * internalNodeChildIndex = m_internalNodeChildIndices->bufferOnDevice();
	void * internalNodeAabbs = m_internalNodeAabbs->bufferOnDevice();
	void * leafNodeAabbs = m_leafAabbs->bufferOnDevice();
	void * mortonCodesAndAabbIndices = m_leafHash[0]->bufferOnDevice();
	void * o_nts = m_ntests->bufferOnDevice();
	bvhRayTraverseIterative((RayInfo *)rays,
								(int *)rootNodeIndex, 
								(int2 *)internalNodeChildIndex, 
								(Aabb *)internalNodeAabbs, 
								(Aabb *)leafNodeAabbs,
								(KeyValuePair *)mortonCodesAndAabbIndices,								
								(float *) o_nts,
								numRays());
}

void BvhSolver::getPoints(BaseBuffer * dst)
{ m_mesh->getVerticesOnDevice(dst); }

void BvhSolver::getRays(BaseBuffer * dst) 
{ m_rays->deviceToHost(dst->data(), m_rays->bufferSize()); }

void BvhSolver::getRootNodeAabb(Aabb * dst)
{ m_internalNodeAabbs->deviceToHost(dst, sizeof(Aabb)); }

void BvhSolver::getLeafAabbs(BaseBuffer * dst)
{ m_leafAabbs->deviceToHost(dst->data(), m_leafAabbs->bufferSize()); }

void BvhSolver::getInternalAabbs(BaseBuffer * dst)
{ m_internalNodeAabbs->deviceToHost(dst->data(), m_internalNodeAabbs->bufferSize()); }

void BvhSolver::getLeafHash(BaseBuffer * dst)
{ m_leafHash[0]->deviceToHost(dst->data(), m_leafHash[0]->bufferSize()); }

void BvhSolver::getInternalDistances(BaseBuffer * dst)
{ m_distanceInternalNodeFromRoot->deviceToHost(dst->data(), m_distanceInternalNodeFromRoot->bufferSize()); }

void BvhSolver::getInternalChildIndex(BaseBuffer * dst)
{ m_internalNodeChildIndices->deviceToHost(dst->data(), m_internalNodeChildIndices->bufferSize()); }

const unsigned BvhSolver::numPoints() const 
{ return m_mesh->numVertices(); }

void BvhSolver::setAlpha(float x) 
{ m_alpha = x; }

const unsigned BvhSolver::numLeafNodes() const 
{ return m_numLeafNodes; }

const unsigned BvhSolver::numInternalNodes() const 
{ return numLeafNodes() - 1; }

const unsigned BvhSolver::numRays() const 
{ return m_numRays; }

void BvhSolver::printLeafInternalNodeConnection()
{
	int * pfl = new int[numInternalNodes()];
	m_internalNodeCommonPrefixLengths->deviceToHost(pfl, m_internalNodeCommonPrefixLengths->bufferSize());
	
	int * lp = new int[numLeafNodes()];
	m_leafNodeParentIndices->deviceToHost(lp,  m_leafNodeParentIndices->bufferSize());
	 
	int2 * cp = new int2[numInternalNodes()];
	m_internalNodeChildIndices->deviceToHost(cp,  m_internalNodeChildIndices->bufferSize());
	
	qDebug()<<"\n leaf - internal node connection:\n";
	int ind = -1;
	for(unsigned i=0; i < numLeafNodes(); i++) {
		if(lp[i] != ind) {
			ind = lp[i];
			qDebug()<<"leaf["<<i<<"] : itl["<<ind<<"] <- ("<<cp[ind].x<<" , "<<cp[ind].y<<") "<<pfl[ind];
		}
		else 
			qDebug()<<"leaf["<<i<<"]";
	}
	delete[] lp;delete[] cp;
}

void BvhSolver::printInternalNodeConnection()
{
	int * ipi = new int[numInternalNodes()];
	m_internalNodeParentIndices->deviceToHost(ipi, m_internalNodeParentIndices->bufferSize());
	
	int2 * cp = new int2[numInternalNodes()];
	m_internalNodeChildIndices->deviceToHost(cp,  m_internalNodeChildIndices->bufferSize());
	
	qDebug()<<"\n internal node connection:\n";
	for(unsigned i=0; i < numInternalNodes(); i++) {
		qDebug()<<ipi[i]<<" <- i["<<i<<"] <- ("<<cp[i].x<<" , "<<cp[i].y<<")";
	}
	
	delete[] cp;
	delete[] ipi;
}

void BvhSolver::getRootNodeIndex(int * dst)
{
	m_rootNodeIndexOnDevice->deviceToHost((void *)dst, m_rootNodeIndexOnDevice->bufferSize());	
}

const bool BvhSolver::isValid() const
{ return m_isValid; }
