/*
 *  BTree.cpp
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "BTree.h"

namespace sdb {

BTree::BTree()
{
	m_root= new BNode<int, Entity>;
}

void BTree::insert(int x)
{
	std::cout<<"insert key "<<x<<"\n";
	Pair<int, Entity> mypair;
    mypair.key = x;
    mypair.index = NULL;
	m_root->insert(mypair);
}

void BTree::remove(int x)
{
	std::cout<<"remove key "<<x<<"\n";
	Pair<int, Entity> mypair;
    mypair.key = x;
    mypair.index = NULL;
	m_root->remove(mypair);
}

void BTree::display()
{
	std::cout<<"\ndisplay tree";
	std::map<int, std::vector<Entity *> > nodes;
	nodes[0].push_back(m_root);
	m_root->getChildren(nodes, 1);
	
	std::map<int, std::vector<Entity *> >::const_iterator it = nodes.begin();
	for(; it != nodes.end(); ++it)
		displayLevel((*it).first, (*it).second);
	std::cout<<"\n";
}

void BTree::displayLevel(const int & level, const std::vector<Entity *> & nodes)
{
	std::cout<<"\n  level: "<<level<<"   ";
	std::vector<Entity *>::const_iterator it = nodes.begin();
	for(; it != nodes.end(); ++it)
		(*it)->display();
}

} // end of namespace sdb