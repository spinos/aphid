/*
 *  KdTreeNodeArray.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "KdTreeNodeArray.h"
#include <iostream>
KdTreeNodeArray::KdTreeNodeArray() 
{
	initialize();
	//std::cout<<"size of kdtreenode "<<sizeof(KdTreeNode)<<"\n";
}

KdTreeNodeArray::~KdTreeNodeArray() 
{
	clear();
}

KdTreeNode *KdTreeNodeArray::asKdTreeNode(unsigned index)
{
	return (KdTreeNode *)at(index);
}

KdTreeNode *KdTreeNodeArray::asKdTreeNode()
{
	return (KdTreeNode *)current();
}

unsigned KdTreeNodeArray::elementSize() const
{ return 8; }

unsigned KdTreeNodeArray::numElementPerBlock() const
{ return 65536; }