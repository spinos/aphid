/*
 *  C3Tree.h
 *  btree
 *
 *  Created by jian zhang on 5/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BNode.h>
#include <List.h>
namespace sdb {

class C3Tree
{
public:
    C3Tree();
    void insert(VertexP & v);
	void remove(VertexP & v);
    void display();
	
	void setGridSize(const float & x);
	
private:
	void displayLevel(const int & level, const std::vector<Entity *> & nodes);
	const Coord3 inGrid(const V3 & p) const;
	float m_gridSize;
	BNode<Coord3, VertexP, List<VertexP> > * m_root;
};
} // end namespace sdb