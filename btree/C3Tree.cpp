/*
 *  C3Tree.cpp
 *  C3Tree
 *
 *  Created by jian zhang on 5/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "C3Tree.h"

namespace sdb {

C3Tree::C3Tree()
{
    TreeNode::MaxNumKeysPerNode = 32;
    TreeNode::MinNumKeysPerNode = 16;
	m_root= new BNode<Coord3, VertexP, List<VertexP> >;
	m_gridSize = 1.f;
}

void C3Tree::setGridSize(const float & x) { m_gridSize = x; }

void C3Tree::insert(VertexP & v)
{
	Coord3 k = inGrid(*v.index);
	
	std::cout<<"insert key "<<k<<"\n";
	Pair<Coord3, VertexP> mypair;
	mypair.key = k;
	mypair.index = &v;
	m_root->insert(mypair);
}

void C3Tree::remove(VertexP & v)
{
	Coord3 k = inGrid(*v.index);
	
	std::cout<<"remove key "<<k<<"\n";
	Pair<Coord3, VertexP> mypair;
	mypair.key = k;
	mypair.index = &v;
	m_root->remove(mypair);
}

void C3Tree::display()
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

void C3Tree::displayLevel(const int & level, const std::vector<Entity *> & nodes)
{
	std::cout<<"\n  level: "<<level<<"   ";
	std::vector<Entity *>::const_iterator it = nodes.begin();
	for(; it != nodes.end(); ++it)
		std::cout<<*(static_cast<BNode<Coord3, VertexP, List<VertexP> > *>(*it));
}

const Coord3 C3Tree::inGrid(const V3 & p) const
{
	Coord3 r;
	r.x = p.data[0] / m_gridSize; if(p.data[0] < 0.f) r.x--;
	r.y = p.data[1] / m_gridSize; if(p.data[1] < 0.f) r.y--;
	r.z = p.data[2] / m_gridSize; if(p.data[2] < 0.f) r.z--;
	return r;
}

} // end namespace sdb