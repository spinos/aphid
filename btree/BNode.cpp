/*
 *  BNode.cpp
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#include "BNode.h"
namespace sdb {
    
int TreeNode::MaxNumKeysPerNode = 8;
int TreeNode::MinNumKeysPerNode = 2;

TreeNode::TreeNode(Entity * parent) : Entity(parent)
{
	m_first = NULL;
	m_isLeaf = false;
}

TreeNode::~TreeNode() 
{
	if(m_first && !isLeaf()) delete m_first;
}

bool TreeNode::isRoot() const 
{ return parent() == NULL; }

bool TreeNode::hasChildren() const 
{ 
	if(isLeaf()) return false; 
	return m_first != NULL;
}

bool TreeNode::isLeaf() const 
{ return m_isLeaf; }

Entity * TreeNode::sibling() const
{ return m_first; }

Entity * TreeNode::firstIndex() const 
{ return m_first; }

void TreeNode::setLeaf() 
{ m_isLeaf = true; }

void TreeNode::connectSibling(Entity * another)
{ m_first = another; }

void TreeNode::setFirstIndex(Entity * another)
{ m_first = another; }

} // end of namespace sdb
//~: