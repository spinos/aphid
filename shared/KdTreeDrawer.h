/*
 *  KdTreeDrawer.h
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BaseDrawer.h>
class KdTree;
class BoundingBox;
class KdTreeNode;
class KdTreeDrawer : public BaseDrawer {
public:
	KdTreeDrawer();
	virtual ~KdTreeDrawer();
	void drawKdTree(const KdTree * tree);
	void drawKdTreeNode(const KdTreeNode * tree, const BoundingBox & bbox);
};
