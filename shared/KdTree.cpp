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
#include <QElapsedTimer>

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
	SplitEvent::Context = &ctx;
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
	
	printf("node sz %d\n", (int)sizeof(KdTreeNode));
	printf("prim sz %d\n", (int)sizeof(Primitive));
	printf("event sz %d\n", (int)sizeof(SplitEvent));
	
}

KdTree::~KdTree() 
{
	delete m_root;
}

KdTreeNode* KdTree::getRoot() const
{ 
	return m_root; 
}

void KdTree::addMesh(BaseMesh* mesh)
{
	unsigned nf = mesh->getNumFaces();
	printf("add %i triangles\n", nf);
	ctx.appendMesh(mesh);
	const BoundingBox box = mesh->calculateBBox();
	m_bbox.expandBy(box);
}

void KdTree::create()
{	
	printf("ctx primitive count %d\n", ctx.getNumPrimitives());
	printf("tree bbox: %f %f %f - %f %f %f\n", m_bbox.m_min.x, m_bbox.m_min.y, m_bbox.m_min.z, m_bbox.m_max.x, m_bbox.m_max.y, m_bbox.m_max.z);
	
	PartitionBound bound;
	bound.bbox = m_bbox;
	bound.parentMin = 0;
	bound.parentMax = ctx.getNumPrimitives();
	
	QElapsedTimer timer;
	timer.start();

	subdivide(m_root, ctx, bound, 0);
	ctx.verbose();
	std::cout << "kd tree finished after " << timer.elapsed() << "ms\n";
}

void KdTree::subdivide(KdTreeNode * node, BuildKdTreeContext & ctx, PartitionBound & bound, int level)
{
	if(bound.numPrimitive() < 64 || level == 18) {
		node->setLeaf(true);
		return;
	}
	
	//printf("subdiv begin %i\n", level);
	
	KdTreeBuilder builder;
	builder.calculateSplitEvents(bound);
	const SplitEvent *plane = builder.bestSplit();
	
	//ctx.verbose();
	
	node->setAxis(plane->getAxis());
	node->setSplitPos(plane->getPos());
	KdTreeNode* branch = ctx.createTreeBranch();
	
	node->setLeft(branch);
	node->setLeaf(false);
	
	ctx.partition(*plane, bound, 1);
	
	BoundingBox leftBox, rightBox;

	bound.bbox.split(plane->getAxis(), plane->getPos(), leftBox, rightBox);
	
	PartitionBound subBound;
	subBound.bbox = leftBox;
	subBound.parentMin = bound.childMin;
	subBound.parentMax = bound.childMax;
	
	if(subBound.numPrimitive() > 0) {
		//printf("ctx partition left %i - %i\n", subBound.parentMin, subBound.parentMax);
		subdivide(branch, ctx, subBound, level + 1);
	}
	
	ctx.releaseIndicesAt(subBound.parentMin);
	
	ctx.partition(*plane, bound, 0);
	
	subBound.bbox = rightBox;
	subBound.parentMin = bound.childMin;
	subBound.parentMax = bound.childMax;
	
	if(subBound.numPrimitive() > 0) {
		//printf("ctx partition right %i - %i\n", rightBound.parentMin, rightBound.parentMax);
		subdivide(branch + 1, ctx, subBound, level + 1);
	}

	ctx.releaseIndicesAt(subBound.parentMin);
	
	//printf("subdiv end %i\n", level);
}
/*
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
*/
