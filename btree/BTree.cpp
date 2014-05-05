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
	m_root= new BNode<int, int, List<int> >;
}

void BTree::insert(int x)
{
	std::cout<<"insert key "<<x<<"\n";
	Pair<int, int> mypair;
    mypair.key = x;
    mypair.index = &x;
	m_root->insert(mypair);
}

void BTree::remove(int x)
{
	std::cout<<"remove key "<<x<<"\n";
	Pair<int, int> mypair;
    mypair.key = x;
    mypair.index = &x;
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
		std::cout<<*(static_cast<BNode<int, int, List<int> > *>(*it));
}

void BTree::displayLeaves()
{
	std::vector<int> d;
	BNode<int, int, List<int> > *l = m_root->firstLeaf();
	std::cout<<"\nall leaves ";
	while(l) {
		std::cout<<(*l);
		l->getValues(d);
		l = static_cast<BNode<int, int, List<int> > *>(l->sibling());
	}
	std::cout<<"\n";
	std::cout<<"\nall values (";
	std::vector<int>::const_iterator it = d.begin();
	for(; it != d.end(); ++it) std::cout<<" "<<*it; 
	std::cout<<" )\n";
}

} // end namespace sdb