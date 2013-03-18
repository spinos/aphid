/*
 *  KdTreeNode.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <APhid.h>
#include "KdTreeNode.h"

KdTreeNode::KdTreeNode() /*: m_combined( 6 )*/ {};

void KdTreeNode::setSplitPos(float a_Pos ) 
{
	inner.split = a_Pos; 
}

float KdTreeNode::getSplitPos() const
{ 
	return inner.split; 
}

void KdTreeNode::setAxis( int a_Axis ) 
{ 
	inner.combined = a_Axis + (inner.combined & EInnerAxisMask); 
}

int KdTreeNode::getAxis() const
{ 
	return inner.combined & (~EInnerAxisMask); 
}

void KdTreeNode::setLeaf( bool a_Leaf ) 
{ 
	leaf.combined = (a_Leaf) ? (leaf.combined | ~ETypeMask):(leaf.combined & ETypeMask); 
}

void KdTreeNode::setPrimStart(unsigned long offset)
{
	leaf.combined = (offset<<3) | ELeafOffsetMask;
}

void KdTreeNode::setNumPrims(unsigned long numPrims)
{
	leaf.end = numPrims;
}

unsigned long KdTreeNode::getPrimStart() const 
{
	return (leaf.combined & ~ELeafOffsetMask) >> 3;
}

unsigned long KdTreeNode::getNumPrims() const 
{
	return leaf.end;
}

bool KdTreeNode::isLeaf() const
{
	return ((leaf.combined & ~ETypeMask) > 0); 
}

void KdTreeNode::setLeft( KdTreeNode* a_Left )
{ 
	inner.combined = (unsigned long)a_Left + (inner.combined & EIndirectionMask); 
}

KdTreeNode* KdTreeNode::getLeft() const
{ 
	return (KdTreeNode*)(inner.combined & ~EIndirectionMask); 
}

KdTreeNode* KdTreeNode::getRight() const 
{ 
	return (KdTreeNode*)(getLeft()) + 1; 
}


//:~