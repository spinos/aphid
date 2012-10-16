/*
 *  KdTreeNode.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "KdTreeNode.h"

KdTreeNode::KdTreeNode() : m_Data( 6 ) {};
void KdTreeNode::setAxis( int a_Axis ) 
{ 
	m_Data = (m_Data & 0xfffffffc) + a_Axis; 
}

int KdTreeNode::getAxis() const
{ 
	return m_Data & 3; 
}

void KdTreeNode::setSplitPos(float a_Pos ) 
{
	m_Split = a_Pos; 
}

float KdTreeNode::getSplitPos() const
{ 
	return m_Split; 
}

void KdTreeNode::setLeft( KdTreeNode* a_Left )
{ 
	m_Data = (unsigned long)a_Left + (m_Data & 7); 
}

KdTreeNode* KdTreeNode::getLeft() const
{ 
	return (KdTreeNode*)(m_Data&0xfffffff8); 
}

KdTreeNode* KdTreeNode::getRight() const 
{ 
	return ((KdTreeNode*)(m_Data&0xfffffff8)) + 1; 
}

bool KdTreeNode::isLeaf() const
{ 
	return ((m_Data & 4) > 0); 
}

void KdTreeNode::setLeaf( bool a_Leaf ) 
{ 
	m_Data = (a_Leaf)?(m_Data|4):(m_Data&0xfffffffb); 
}
//:~