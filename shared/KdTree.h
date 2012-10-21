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
#include <KdTreeBuilder.h>

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

	BoundingBox m_bbox;
	
private:	
	KdTreeBuilder m_builder;
	BuildKdTreeContext ctx;
	KdTreeNode * m_root;
};