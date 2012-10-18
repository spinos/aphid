/*
 *  KdTreeNode.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "KdTreeNode.h"

KdTreeNode::KdTreeNode() : m_combined( 6 ) {};

void KdTreeNode::setSplitPos(float a_Pos ) 
{
	m_Split = a_Pos; 
}

float KdTreeNode::getSplitPos() const
{ 
	return m_Split; 
}

void KdTreeNode::setAxis( int a_Axis ) 
{ 
	m_combined = a_Axis + (m_combined & EInnerAxisMask); 
}

int KdTreeNode::getAxis() const
{ 
	return m_combined & (~EInnerAxisMask); 
}

void KdTreeNode::setLeaf( bool a_Leaf ) 
{ 
	m_combined = (a_Leaf) ? (m_combined | ~ETypeMask):(m_combined & ETypeMask); 
}

bool KdTreeNode::isLeaf() const
{ 
	return ((m_combined & ~ETypeMask) > 0); 
}

void KdTreeNode::setLeft( KdTreeNode* a_Left )
{ 
	m_combined = (unsigned long)a_Left + (m_combined & EIndirectionMask); 
}

KdTreeNode* KdTreeNode::getLeft() const
{ 
	return (KdTreeNode*)(m_combined & ~EIndirectionMask); 
}

KdTreeNode* KdTreeNode::getRight() const 
{ 
	return (KdTreeNode*)(getLeft()) + 1; 
}


//:~