/*
 *  KdTree.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "KdTree.h"

const char *byte_to_binary(int x)
{
    static char b[33];
    b[32] = '\0';

    for (int z = 0; z < 32; z++) {
        b[31-z] = ((x>>z) & 0x1) ? '1' : '0';
    }

    return b;

}

KdTree::KdTree() 
{
	m_root = new KdTreeNode;
	
	printf("axis mask        %s\n", byte_to_binary(KdTreeNode::EInnerAxisMask));
	printf("type        mask %s\n", byte_to_binary(KdTreeNode::ETypeMask));
	printf("indirection mask %s\n", byte_to_binary(KdTreeNode::EIndirectionMask));
	
	printf("32 bit align mask %s\n", byte_to_binary(0xffffffff - 31));
	
	unsigned lc = 127;
	lc = lc & (0xffffffff - 31);
	printf("32 bit align lc %d\n", lc);
	printf("32 bit align lc %s\n", byte_to_binary(lc));
	printf("2199              %s\n", byte_to_binary(2199));

	printf("31              %s\n", byte_to_binary(31));
	
	printf("2199 & 31         %s\n", byte_to_binary(2199 & 31));
	printf("2199 & 31         %d\n", 2199 & 31);
	

}

KdTree::~KdTree() 
{
	delete m_root;
}

KdTreeNode* KdTree::getRoot() const
{ 
	return m_root; 
}

void KdTree::create(BaseMesh* mesh)
{
	BuildKdTreeContext ctx;
	ctx.appendMesh(mesh);
	ctx.initIndices();
	printf("ctx primitive count %d\n", ctx.getNumPrimitives());
	BoundingBox bbox = ctx.calculateTightBBox();
	printf("ctx tight bbox: %f %f %f - %f %f %f\n", bbox.m_min.x, bbox.m_min.y, bbox.m_min.z, bbox.m_max.x, bbox.m_max.y, bbox.m_max.z);
	m_bbox.setMin(-3.f, -3.f, -3.f);
	m_bbox.setMax(35.f, 35.f, 36.f);
	ctx.setBBox(m_bbox);
	
	printf("node sz %d\n", (int)sizeof(KdTreeNode));
	printf("prim sz %d\n", (int)sizeof(Primitive));
	
	unsigned nf = mesh->getNumFaces();
	printf("num triangles %i \n", nf);
	allocateTree(nf * 2 + 1);
	subdivide(m_root, ctx);
	
	
/*
	primitivePtr * primitives = new primitivePtr[nf];
	for(unsigned i = 0; i < nf; i++) {
		primitives[i]->setGeom((char*)mesh->getFace(i));
		primitives[i]->setType(0);
	}
	
	subdivide(m_root, primitives, m_bbox, 0, nf - 1);
		
	for(unsigned i = 0; i < nf; i++) delete primitives[i];
	delete[] primitives;*/
}

void KdTree::allocateTree(unsigned num)
{
	m_nodePtr = new char[num * sizeof(KdTreeNode) + 31];
	m_currentNode = (KdTreeNode *)(((unsigned long)m_nodePtr + 32) & (0xffffffff - 31));
}

void KdTree::subdivide(KdTreeNode * node, BuildKdTreeContext & ctx)
{
	if(ctx.getNumPrimitives() < 64) {
		node->setLeaf(true);
		return;
	}
	
	SplitCandidate plane = ctx.bestSplit();
	plane.verbose();
	ctx.partition(plane);
	ctx.verbose();
	
	node->setAxis(plane.getAxis());
	node->setSplitPos(plane.getPos());
	KdTreeNode* branch = treeNodePair();
	
	node->setLeft(branch);
	node->setLeaf(false);
	
	BoundingBox leftBox, rightBox;
	
	ctx.getBBox().split(plane.getAxis(), plane.getPos(), leftBox, rightBox);
	
	BuildKdTreeContext leftCtx;
	leftCtx.setPrimitives(ctx.getPrimitives());
	leftCtx.setIndices(ctx.getLeftIndices());
	leftCtx.setBBox(leftBox);
	
	
	BuildKdTreeContext rightCtx;
	rightCtx.setPrimitives(ctx.getPrimitives());
	rightCtx.setIndices(ctx.getRightIndices());
	rightCtx.setBBox(rightBox);
	
	subdivide(branch, leftCtx);
	subdivide(branch + 1, rightCtx);
}

void KdTree::subdivide(KdTreeNode * node, primitivePtr * prim, BoundingBox bbox, unsigned first, unsigned last)
{
	//printf("first last: %i %i\n", first, last);
	if(first == last) {
		node->setLeaf(true);
		return;
	}
	
	//printf("bbox: %f %f %f - %f %f %f\n", bbox.m_min.x, bbox.m_min.y, bbox.m_min.z, bbox.m_max.x, bbox.m_max.y, bbox.m_max.z);
	
	int axis = bbox.getLongestAxis();
	
	//printf("axis: %i\n", axis);
	
	node->setAxis(axis);
	
	sort(prim, first, last, axis);
	
	float splitPos = ((Triangle *)prim[(first + last)/2])->getMin(axis);
	
	//printf("split at %f\n", splitPos);
	
	node->setSplitPos(splitPos);
	
	KdTreeNode* branch = treeNodePair();
	
	node->setLeft(branch);
	node->setLeaf(false);
	
	BoundingBox leftBox, rightBox;
	
	bbox.split(axis, splitPos, leftBox, rightBox);
	
	subdivide(branch, prim, leftBox, first, (first + last)/2);
	if(last >= (first + last)/2 + 1) subdivide(branch + 1, prim, rightBox, (first + last)/2 + 1, last);
}

KdTreeNode* KdTree::treeNodePair()
{ 
	unsigned long * tmp = (unsigned long*)m_currentNode;
	tmp[1] = tmp[3] = 6;
	KdTreeNode* node = m_currentNode;
	m_currentNode += 2;
	return node;
}

void KdTree::sort(primitivePtr * prim, unsigned first, unsigned last, int axis)
{
	if(last <= first) return;
	
	primitivePtr temp;
	
	if(last == first + 1) {
		if(((Triangle *)prim[first])->getMin(axis) > ((Triangle *)prim[last])->getMin(axis)) {
			temp = prim[last];
			prim[last] = prim[first];
			prim[first] = temp;
		}
		return;
	}
	
	unsigned low,high;
	float list_separator;

	low = first;
	high = last;
	
	primitivePtr mid = prim[(first+last)/2];
	
	if(mid->getType() == 0)
		list_separator = ((Triangle *)mid)->getMin(axis);

	do {
		while(((Triangle *)prim[low])->getMin(axis) < list_separator) low++;
		while(((Triangle *)prim[high])->getMin(axis) > list_separator) high--;

		if(low<=high)
		{
			temp = prim[low];
			prim[low++] = prim[high];
			prim[high--] = temp;
		}
	} while(low<=high);
	if(first<high) sort(prim,first,high, axis);
	if(low<last) sort(prim,low,last, axis);
}

