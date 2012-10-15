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

class KdTree
{
public:
	KdTree();
	~KdTree();
	
	KdTreeNode* GetRoot();
	
	void create(BaseMesh* mesh);

	KdTreeNode* m_root;
};