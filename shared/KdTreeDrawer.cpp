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
	beginQuad();
	drawKdTreeNode(root, bbox);
	end();
}

void KdTreeDrawer::drawKdTreeNode(const KdTreeNode * tree, const BoundingBox & bbox)
{
	Vector3F corner0(bbox.m_min_x, bbox.m_min_y, bbox.m_min_z);
	Vector3F corner1(bbox.m_max_x, bbox.m_max_y, bbox.m_max_z);
	if(tree->isLeaf()) return;

	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	
	glVertex3f(corner0.x, corner1.y, corner0.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	
	int axis = tree->getAxis();
	
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
	
	BoundingBox leftBox, rightBox;
	
	float splitPos = tree->getSplitPos();
	bbox.split(axis, splitPos, leftBox, rightBox);
	
	drawKdTreeNode(tree->getLeft(), leftBox);
	drawKdTreeNode(tree->getRight(), rightBox);
	
}
