/*
 *  KdTreeDrawer.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif
#include "KdTreeDrawer.h"
#include <KdTree.h>

KdTreeDrawer::KdTreeDrawer() {}
KdTreeDrawer::~KdTreeDrawer() {}

void KdTreeDrawer::drawKdTree(const KdTree * tree)
{
	BoundingBox bbox = tree->m_bbox;
	KdTreeNode * root = tree->getRoot();
	
	setWired(1);
	int level = 0;
	drawKdTreeNode(root, bbox, level);
}

void KdTreeDrawer::drawKdTreeNode(const KdTreeNode * tree, const BoundingBox & bbox, int level)
{
	if(level > 17) return;
	if(tree->isLeaf()) return;
	setColor(.5f, 0.1f, 0.f);
	boundingBox(bbox);
	Vector3F corner0(bbox.getMin(0), bbox.getMin(1), bbox.getMin(2));
	Vector3F corner1(bbox.getMax(0), bbox.getMax(1), bbox.getMax(2));

	int axis = tree->getAxis();
	setColor(1.f, 1.f, 0.f);
	glBegin(GL_LINE_LOOP);
	if(axis == 0) {
		corner0.x = corner1.x = tree->getSplitPos();
		glVertex3f(corner0.x, corner0.y, corner0.z);
		glVertex3f(corner0.x, corner1.y, corner0.z);
		glVertex3f(corner0.x, corner1.y, corner1.z);
		glVertex3f(corner0.x, corner0.y, corner1.z);
	}
	else if(axis == 1) {
		corner0.y = corner1.y = tree->getSplitPos();
		glVertex3f(corner0.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner0.y, corner1.z);
		glVertex3f(corner0.x, corner0.y, corner1.z);
	}
	else {
		corner0.z = corner1.z = tree->getSplitPos();
		glVertex3f(corner0.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner1.y, corner0.z);
		glVertex3f(corner0.x, corner1.y, corner0.z);
	}
	glEnd();
	BoundingBox leftBox, rightBox;
	
	float splitPos = tree->getSplitPos();
	bbox.split(axis, splitPos, leftBox, rightBox);
	level++;
	drawKdTreeNode(tree->getLeft(), leftBox, level);
	drawKdTreeNode(tree->getRight(), rightBox, level);
	
}

void KdTreeDrawer::drawPrimitivesInNode(KdTree * tree, const KdTreeNode * node)
{
	unsigned start = node->getPrimStart();
	unsigned num = node->getNumPrims();
	for(unsigned i = 0; i < num; i++) {
	    Primitive * prim = tree->getPrim(start + i);
	    primitive(prim);
	}
}
