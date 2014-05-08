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
#include <BoundingBox.h>
namespace sdb {

class C3Tree
{
public:
    C3Tree();
    void insert(VertexP & v);
	void remove(VertexP & v);
    void display();
	
	void setGridSize(const float & x);
	
	void firstGrid();
	void nextGrid();
	const bool gridEnd() const;
	const BoundingBox gridBoundingBox() const;
	const BoundingBox boundingBox() const;
	void calculateBBox();
	const List<VertexP> * verticesInGrid() const;
private:
	typedef BNode<Coord3> C3NodeType;

	void displayLevel(const int & level, const std::vector<Entity *> & nodes);
	const Coord3 inGrid(const V3 & p) const;
	const BoundingBox coordToGridBBox(const Coord3 & c) const;
	void updateBBox(const BoundingBox & b);
	
	void firstLeaf();
	void nextLeaf();
	const bool leafEnd() const;
	
	BoundingBox m_bbox;
	C3NodeType * m_root;
	C3NodeType * m_current;
	float m_gridSize;
	int m_currentData, m_dataEnd;
};
} // end namespace sdb