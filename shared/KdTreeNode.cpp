/*
 *  KdTreeNode.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <stdint.h>
#include "KdTreeNode.h"
namespace aphid {

KdTreeNode::KdTreeNode() {};

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
	inner.combined = a_Axis | (inner.combined & EInnerAxisMask); 
}

int KdTreeNode::getAxis() const
{ 
	return inner.combined & (~EInnerAxisMask); 
}

void KdTreeNode::setLeaf() 
{ leaf.combined = ETypeMaskTau; }

void KdTreeNode::setInternal()
{ leaf.combined = leaf.combined & ETypeMask; }

void KdTreeNode::setPrimStart(unsigned long offset)
{
	leaf.combined = (offset<<3) | ELeafOffsetMask;
}

void KdTreeNode::setNumPrims(unsigned long numPrims)
{
	leaf.end = numPrims;
}

int KdTreeNode::getPrimStart() const 
{
	return (leaf.combined & (~ELeafOffsetMask) ) >> 3;
}

int KdTreeNode::getNumPrims() const 
{
	return leaf.end;
}

bool KdTreeNode::isLeaf() const
{
	return ((leaf.combined & (~ETypeMask) ) > 0); 
}

void KdTreeNode::setLeft( KdTreeNode* a_Left )
{ 
    uintptr_t rawR = reinterpret_cast<uintptr_t>(this);
	uintptr_t rawInt = reinterpret_cast<uintptr_t>(a_Left);
	inner.combined = (rawInt - rawR) | (inner.combined & EIndirectionMask); 
}

KdTreeNode* KdTreeNode::getLeft() const
{ 
    uintptr_t rawR = reinterpret_cast<uintptr_t>(this);
	return (KdTreeNode*)(rawR + (inner.combined & ~EIndirectionMask) ); 
}

KdTreeNode* KdTreeNode::getRight() const 
{ 
	return (KdTreeNode*)(getLeft()) + 1; 
}

void KdTreeNode::setOffset(int x)
{ inner.combined = (x<<3) + (inner.combined & EIndirectionMask); }

int KdTreeNode::getOffset() const
{ return (inner.combined & ~EIndirectionMask)>>3; }

}
//:~