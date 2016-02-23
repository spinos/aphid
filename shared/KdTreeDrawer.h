/*
 *  KdTreeDrawer.h
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <GeoDrawer.h>

namespace aphid {

class KdTree;
class BoundingBox;
class KdTreeNode;

class KdTreeDrawer {
public:
	KdTreeDrawer();
	void drawKdTree(KdTree * tree);
	void drawKdTreeNode(KdTreeNode * tree, const BoundingBox & bbox, int level);
};

}