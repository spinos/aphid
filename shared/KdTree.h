/*
 *  KdTree.h
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <KdTreeNode.h>
#include <BaseMesh.h>
#include <BoundingBox.h>
#include <Primitive.h>
#include <BuildKdTreeContext.h>

typedef Primitive * primitivePtr;
	
class KdTree
{
public:
	KdTree();
	~KdTree();
	
	KdTreeNode* getRoot() const;
	
	void addMesh(BaseMesh* mesh);
	void create();
	
	void subdivide(KdTreeNode * node, BuildKdTreeContext & ctx, PartitionBound & bound, int level);
	
	BuildKdTreeContext ctx;
	BoundingBox m_bbox;
	KdTreeNode * m_root;
};