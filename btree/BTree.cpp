/*
 *  BTree.cpp
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "BTree.h"

BTree::BTree()
{
	m_root= new BNode;
}

void BTree::insert(int x)
{
	std::cout<<"\ninsert key "<<x<<"\n";
	BNode::Pair mypair;
    mypair.key = x;
    mypair.index = NULL;
	m_root->insert(mypair);
}

void BTree::remove(int x)
{
	std::cout<<"\nremove key "<<x<<"\n";
	BNode::Pair mypair;
    mypair.key = x;
    mypair.index = NULL;
	m_root->remove(mypair);
}

void BTree::display()
{
	std::cout<<"\ndisplay tree";
	BTreeDisplayMap nodes;
	nodes[0].push_back(m_root);
	m_root->getChildren(nodes, 1);
	
	BTreeDisplayMap::const_iterator it = nodes.begin();
	for(; it != nodes.end(); ++it)
		displayLevel((*it).first, (*it).second);
	std::cout<<"\n";
}

void BTree::displayLevel(const int & level, const std::vector<BNode *> & nodes)
{
	std::cout<<"\n  level: "<<level<<"   ";
	std::vector<BNode *>::const_iterator it = nodes.begin();
	for(; it != nodes.end(); ++it)
		(*it)->display();
}