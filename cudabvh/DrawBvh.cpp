/*
 *  DrawBvh.cpp
 *  cudabvh
 *
 *  Created by jian zhang on 2/18/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawBvh.h"
#include <GeoDrawer.h>
#include "CudaLinearBvh.h"
#include "CudaBroadphase.h"
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include "bvh_common.h"
#include <radixsort_implement.h>
#include <stripedModel.h>

DrawBvh::DrawBvh() 
{
	m_displayLeafAabbs = new BaseBuffer;
	m_displayInternalAabbs = new BaseBuffer;
	m_displayLeafHash = new BaseBuffer;
	m_displayInternalDistance = new BaseBuffer;
	m_internalChildIndices = new BaseBuffer;
	m_pairCounts = new BaseBuffer;
	m_scanCounts = new BaseBuffer;
	m_pairCache = new BaseBuffer;
	m_boxes = new BaseBuffer;
	m_uniquePairs = new BaseBuffer;
	m_scanUniquePairs = new BaseBuffer;
	m_displayLevel = 12;
}

DrawBvh::~DrawBvh() {}

void DrawBvh::setDrawer(GeoDrawer * drawer)
{ m_drawer = drawer; }

void DrawBvh::setBvh(CudaLinearBvh * bvh)
{ m_bvh = bvh; }

void DrawBvh::addDispalyLevel()
{ m_displayLevel++; }

void DrawBvh::minusDispalyLevel()
{ m_displayLevel--; }

void DrawBvh::showBound(CudaLinearBvh * bvh)
{
	Aabb ab = bvh->bound();
	BoundingBox bb;
	bb.setMin(ab.low.x, ab.low.y, ab.low.z);
	bb.setMax(ab.high.x, ab.high.y, ab.high.z);
	
	m_drawer->boundingBox(bb);
}

void DrawBvh::leaf()
{
	const unsigned n = m_bvh->numLeafNodes();
	m_displayLeafAabbs->create(n * sizeof(Aabb));
	m_bvh->getLeafAabbs(m_displayLeafAabbs);
	Aabb * boxes = (Aabb *)m_displayLeafAabbs->data();
	BoundingBox bb;
	for(unsigned i=0; i < n; i++) {
		Aabb & ab = boxes[i];
		bb.setMin(ab.low.x, ab.low.y, ab.low.z);
		bb.setMax(ab.high.x, ab.high.y, ab.high.z);
		m_drawer->boundingBox(bb);
	}
}

void DrawBvh::showHash(CudaLinearBvh * bvh)
{
	const unsigned n = bvh->numLeafNodes();
	
	m_displayLeafAabbs->create(n * sizeof(Aabb));
	bvh->getLeafAabbs(m_displayLeafAabbs);
	Aabb * boxes = (Aabb *)m_displayLeafAabbs->data();
	
	m_displayLeafHash->create(n * sizeof(KeyValuePair));
	bvh->getLeafHash(m_displayLeafHash);
	KeyValuePair * leafHash = (KeyValuePair *)m_displayLeafHash->data();
	
	const unsigned numInternal = bvh->numInternalNodes();
	glBegin(GL_LINES);
	for(unsigned i=0; i < numInternal; i++) {
		float red = (float)i/(float)numInternal;
		
		glColor3f(red, 1.f - red, 0.f);
		Aabb a0 = boxes[leafHash[i].value];
		glVertex3f(a0.low.x * 0.5f + a0.high.x * 0.5f, a0.low.y * 0.5f + a0.high.y * 0.5f + 0.2f, a0.low.z * 0.5f + a0.high.z * 0.5f);
        
		Aabb a1 = boxes[leafHash[i+1].value];
		glVertex3f(a1.low.x * 0.5f + a1.high.x * 0.5f, a1.low.y * 0.5f + a1.high.y * 0.5f + 0.2f, a1.low.z * 0.5f + a1.high.z * 0.5f);
        
	}
	glEnd();
}

inline int isLeafNode(int index) 
{ return (index >> 31 == 0); }

inline int getIndexWithInternalNodeMarkerRemoved(int index) 
{ return index & (~0x80000000); }

void DrawBvh::showHierarch(CudaLinearBvh * bvh)
{
	bvh->getRootNodeIndex(&m_hostRootNodeInd);
	
	const unsigned numInternal = bvh->numInternalNodes();
	
	m_displayInternalAabbs->create(numInternal * sizeof(Aabb));
	bvh->getInternalAabbs(m_displayInternalAabbs);
	Aabb * internalBoxes = (Aabb *)m_displayInternalAabbs->data();
	
	m_displayLeafHash->create(bvh->numLeafNodes() * sizeof(KeyValuePair));
	bvh->getLeafHash(m_displayLeafHash);
	KeyValuePair * leafHash = (KeyValuePair *)m_displayLeafHash->data();
	
	m_displayLeafAabbs->create(bvh->numLeafNodes() * sizeof(Aabb));
	bvh->getLeafAabbs(m_displayLeafAabbs);
	Aabb * leafBoxes = (Aabb *)m_displayLeafAabbs->data();
	
	m_internalChildIndices->create(bvh->numInternalNodes() * sizeof(int2));
	bvh->getInternalChildIndex(m_internalChildIndices);
	int2 * internalNodeChildIndices = (int2 *)m_internalChildIndices->data();
	
	m_displayInternalDistance->create(bvh->numInternalNodes() * sizeof(int));
	bvh->getInternalDistances(m_displayInternalDistance);
	int * levels = (int *)m_displayInternalDistance->data();
	
	BoundingBox bb;
	
	int stack[128];
	stack[0] = m_hostRootNodeInd;
	int stackSize = 1;
	int maxStack = 1;
	int touchedLeaf = 0;
	int touchedInternal = 0;
	while(stackSize > 0) {
		int internalOrLeafNodeIndex = stack[ stackSize - 1 ];
		stackSize--;
		
		int isLeaf = isLeafNode(internalOrLeafNodeIndex);	//Internal node if false
		uint bvhNodeIndex = getIndexWithInternalNodeMarkerRemoved(internalOrLeafNodeIndex);
		
		int bvhRigidIndex = (isLeaf) ? leafHash[bvhNodeIndex].value : -1;
		
		Aabb bvhNodeAabb = (isLeaf) ? leafBoxes[bvhRigidIndex] : internalBoxes[bvhNodeIndex];

		{
			if(isLeaf) {
				glColor3f(.5, 0., 0.);
				bb.setMin(bvhNodeAabb.low.x, bvhNodeAabb.low.y, bvhNodeAabb.low.z);
				bb.setMax(bvhNodeAabb.high.x, bvhNodeAabb.high.y, bvhNodeAabb.high.z);
				m_drawer->boundingBox(bb);

				touchedLeaf++;
			}
			else {
				glColor3f(.5, .65, 0.);
				
				if(levels[bvhNodeIndex] > m_displayLevel) continue;
				bb.setMin(bvhNodeAabb.low.x, bvhNodeAabb.low.y, bvhNodeAabb.low.z);
				bb.setMax(bvhNodeAabb.high.x, bvhNodeAabb.high.y, bvhNodeAabb.high.z);
				m_drawer->boundingBox(bb);

				touchedInternal++;
				if(stackSize + 2 > 128)
				{
					//Error
				}
				else
				{
				    stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].x;
					stackSize++;
					stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].y;
					stackSize++;
					
					if(stackSize > maxStack) maxStack = stackSize;
				}
			}
		}
	}
	
}

void DrawBvh::printHash()
{
    m_displayLeafHash->create(m_bvh->numLeafNodes() * sizeof(KeyValuePair));
	m_bvh->getLeafHash(m_displayLeafHash);
	KeyValuePair * leafHash = (KeyValuePair *)m_displayLeafHash->data();
	for(unsigned i=0; i< m_bvh->numLeafNodes(); i++)
	    std::cout<<" "<<i<<" "<<leafHash[i].key<<" "<<leafHash[i].value<<" ";
}

void DrawBvh::printPairCounts(CudaBroadphase * broadphase)
{
	const unsigned nb = broadphase->numBoxes();
	m_pairCounts->create(nb * sizeof(unsigned));
	broadphase->getOverlappingPairCounts(m_pairCounts);
	
	m_scanCounts->create(nb * sizeof(unsigned));
	broadphase->getScanCounts(m_scanCounts);
	
	unsigned * count = (unsigned *)m_pairCounts->data();
	unsigned * sum = (unsigned *)m_scanCounts->data();
	
	unsigned i;
	for(i=0; i < nb; i++)
		std::cout<<" "<<i<<": "<<count[i]<<" "<<sum[i]<<" ";
	
	std::cout<<" "<<count[nb - 1]+sum[nb - 1]<<"overlapping pairs";
	
	const unsigned cacheLength = broadphase->pairCacheLength();
	if(cacheLength < 1) return;
	m_pairCache->create(cacheLength * 8);
	broadphase->getOverlappingPairCache(m_pairCache);
	m_uniquePairs->create(cacheLength * 4);
	broadphase->getUniquePairs(m_uniquePairs);
	
	m_scanUniquePairs->create(cacheLength * 4);
	broadphase->getScanUniquePairs(m_scanUniquePairs);
	
	unsigned * pc = (unsigned *)m_pairCache->data();
	unsigned * u = (unsigned *)m_uniquePairs->data();
	unsigned * s = (unsigned *)m_scanUniquePairs->data();
	
	for(i=0; i < cacheLength; i++)
	    std::cout<<" "<<i<<": "<<pc[i * 2]<<" "<<pc[i * 2 + 1]<<" is "<<u[i]<<" "<<s[i]<<" \n";
}

void DrawBvh::showOverlappingPairs(CudaBroadphase * broadphase)
{
    const unsigned cacheLength = broadphase->pairCacheLength();
	if(cacheLength < 1) return;
	
	glDisable(GL_DEPTH_TEST);
	
	// std::cout<<" num overlapping pairs "<<cacheLength<<" squeezed to "<<broadphase->numUniquePairs()<<"\n";
	
	const unsigned nb = broadphase->numBoxes();
	m_boxes->create(nb * 24);
	
	broadphase->getBoxes(m_boxes);

	Aabb * boxes = (Aabb *)m_boxes->data();
	Aabb abox;
	BoundingBox ab, bb;
	unsigned i;
	m_drawer->setColor(0.f, 0.1f, 0.3f);

	m_pairCache->create(broadphase->numUniquePairs() * 8);
	CUDABuffer * uniquePairs = broadphase->overlappingPairBuf();
	uniquePairs->deviceToHost(m_pairCache->data(), m_pairCache->bufferSize());
	unsigned * pc = (unsigned *)m_pairCache->data();
	
	unsigned objectI;
	for(i=0; i < broadphase->numUniquePairs(); i++) {
	    objectI = extractObjectInd(pc[i * 2]);
	    abox = boxes[broadphase->objectStart(objectI) + extractElementInd(pc[i * 2])];
	    
		bb.setMin(abox.low.x, abox.low.y, abox.low.z);
		bb.setMax(abox.high.x, abox.high.y, abox.high.z);
	    
	    objectI = extractObjectInd(pc[i * 2 + 1]);
	    abox = boxes[broadphase->objectStart(objectI) + extractElementInd(pc[i * 2 + 1])];
	    
	    ab.setMin(abox.low.x, abox.low.y, abox.low.z);
		ab.setMax(abox.high.x, abox.high.y, abox.high.z);
		
		m_drawer->arrow(bb.center(), ab.center());
		
		bb.expandBy(ab);
		
		// m_drawer->boundingBox(bb);
	}
	glEnable(GL_DEPTH_TEST);
}

