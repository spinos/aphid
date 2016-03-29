/*
 *  BNode.cpp
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#include "BNode.h"
namespace aphid {

namespace sdb {
    
int TreeNode::MaxNumKeysPerNode = 128;
int TreeNode::MinNumKeysPerNode = 16;

TreeNode::TreeNode(Entity * parent) : Entity(parent)
{
	m_link = NULL;
	m_isLeaf = false;
}

TreeNode::~TreeNode() 
{
	if(m_link && !isLeaf()) delete m_link;
}

bool TreeNode::isRoot() const 
{ return parent() == NULL; }

bool TreeNode::hasChildren() const 
{ 
	if(isLeaf()) return false; 
	return m_link != NULL;
}

bool TreeNode::isLeaf() const 
{ return m_isLeaf; }

Entity * TreeNode::sibling() const
{ return m_link; }

Entity * TreeNode::leftChild() const 
{ return m_link; }

void TreeNode::setLeaf() 
{ m_isLeaf = true; }

void TreeNode::connectSibling(Entity * another)
{ m_link = another; }

void TreeNode::connectLeftChild(Entity * another)
{ m_link = another; }

} // end of namespace sdb

}
//~: