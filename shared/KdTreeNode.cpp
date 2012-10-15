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
void KdTreeNode::SetAxis( int a_Axis ) 
{ 
	m_Data = (m_Data & 0xfffffffc) + a_Axis; 
}

int KdTreeNode::GetAxis() 
{ 
	return m_Data & 3; 
}

void KdTreeNode::SetSplitPos(float a_Pos ) 
{
	m_Split = a_Pos; 
}

float KdTreeNode::GetSplitPos() 
{ 
	return m_Split; 
}

void KdTreeNode::SetLeft( KdTreeNode* a_Left ) 
{ 
	m_Data = (unsigned long)a_Left + (m_Data & 7); 
}

KdTreeNode* KdTreeNode::GetLeft()
{ 
	return (KdTreeNode*)(m_Data&0xfffffff8); 
}

KdTreeNode* KdTreeNode::GetRight() 
{ 
	return ((KdTreeNode*)(m_Data&0xfffffff8)) + 1; 
}

bool KdTreeNode::IsLeaf() 
{ 
	return ((m_Data & 4) > 0); 
}

void KdTreeNode::SetLeaf( bool a_Leaf ) 
{ 
	m_Data = (a_Leaf)?(m_Data|4):(m_Data&0xfffffffb); 
}
//:~