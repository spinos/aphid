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

namespace aphid {

KdTreeDrawer::KdTreeDrawer() {}

void KdTreeDrawer::drawKdTree(KdTree * tree)
{
	if(tree->isEmpty()) return;
	BoundingBox bbox = tree->getBBox();
	KdTreeNode * root = tree->getRoot();
	
	glColor3f(0.f, 0.3f, 0.1f);
	
	int level = 0;
	drawKdTreeNode(root, bbox, level);
}

void KdTreeDrawer::drawKdTreeNode(KdTreeNode * tree, const BoundingBox & bbox, int level)
{
	if(level == 32) return;
	if(tree->isLeaf()) return;
	
	Vector3F corner0(bbox.getMin(0), bbox.getMin(1), bbox.getMin(2));
	Vector3F corner1(bbox.getMax(0), bbox.getMax(1), bbox.getMax(2));
	const int axis = tree->getAxis();
	corner0.setComp(tree->getSplitPos(), axis);
	corner1.setComp(tree->getSplitPos(), axis);
	
	glBegin(GL_LINE_LOOP);
	if(axis == 0) {
		glVertex3f(corner0.x, corner0.y, corner0.z);
		glVertex3f(corner0.x, corner1.y, corner0.z);
		glVertex3f(corner0.x, corner1.y, corner1.z);
		glVertex3f(corner0.x, corner0.y, corner1.z);
	}
	else if(axis == 1) {
		glVertex3f(corner0.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner0.y, corner1.z);
		glVertex3f(corner0.x, corner0.y, corner1.z);
	}
	else {
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

}