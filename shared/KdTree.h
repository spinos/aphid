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
	
	void create(BaseMesh* mesh);
	
	void allocateTree(unsigned num);
	void subdivide(KdTreeNode * node, BuildKdTreeContext & ctx);

	void subdivide(KdTreeNode * node, primitivePtr * prim, BoundingBox bbox, unsigned first, unsigned last);
	
	KdTreeNode* treeNodePair();
	
	void sort(primitivePtr * prim, unsigned first, unsigned last, int axis);

	BoundingBox m_bbox;
	KdTreeNode * m_root;
	char * m_nodePtr;
	KdTreeNode * m_currentNode;
	
};