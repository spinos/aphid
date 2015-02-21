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
#include <BaseBuffer.h>
#include "bvh_common.h"
#include <radixsort_implement.h>

DrawBvh::DrawBvh() 
{
	m_displayLeafAabbs = new BaseBuffer;
	m_displayInternalAabbs = new BaseBuffer;
	m_displayLeafHash = new BaseBuffer;
	m_displayInternalDistance = new BaseBuffer;
	m_internalChildIndices = new BaseBuffer;
	m_displayLevel = 3;
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

void DrawBvh::bound()
{
	Aabb ab = m_bvh->bound();
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

void DrawBvh::hash()
{
	const unsigned n = m_bvh->numLeafNodes();
	
	m_displayLeafAabbs->create(n * sizeof(Aabb));
	m_bvh->getLeafAabbs(m_displayLeafAabbs);
	Aabb * boxes = (Aabb *)m_displayLeafAabbs->data();
	
	m_displayLeafHash->create(n * sizeof(KeyValuePair));
	m_bvh->getLeafHash(m_displayLeafHash);
	KeyValuePair * leafHash = (KeyValuePair *)m_displayLeafHash->data();
	
	const unsigned numInternal = m_bvh->numInternalNodes();
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

void DrawBvh::hierarch()
{
	m_bvh->getRootNodeIndex(&m_hostRootNodeInd);
	
	const unsigned numInternal = m_bvh->numInternalNodes();
	
	m_displayInternalAabbs->create(numInternal * sizeof(Aabb));
	m_bvh->getInternalAabbs(m_displayInternalAabbs);
	Aabb * internalBoxes = (Aabb *)m_displayInternalAabbs->data();
	
	m_displayLeafHash->create(m_bvh->numLeafNodes() * sizeof(KeyValuePair));
	m_bvh->getLeafHash(m_displayLeafHash);
	KeyValuePair * leafHash = (KeyValuePair *)m_displayLeafHash->data();
	
	m_displayLeafAabbs->create(m_bvh->numLeafNodes() * sizeof(Aabb));
	m_bvh->getLeafAabbs(m_displayLeafAabbs);
	Aabb * leafBoxes = (Aabb *)m_displayLeafAabbs->data();
	
	m_internalChildIndices->create(m_bvh->numInternalNodes() * sizeof(int2));
	m_bvh->getInternalChildIndex(m_internalChildIndices);
	int2 * internalNodeChildIndices = (int2 *)m_internalChildIndices->data();
	
	m_displayInternalDistance->create(m_bvh->numInternalNodes() * sizeof(int));
	m_bvh->getInternalDistances(m_displayInternalDistance);
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

