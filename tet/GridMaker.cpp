/*
 *  GridMaker.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "GridMaker.h"
#include <iostream>

using namespace aphid;

namespace ttg {
GridMaker::GridMaker()
{}

GridMaker::~GridMaker()
{ 
	m_grid.clear(); 
	std::vector<ITetrahedron *>::iterator it = m_tets.begin();
	for(;it!=m_tets.end();++it) {
		delete *it;
	}
	m_tets.clear();
	m_graphEdges.clear();
	
}

void GridMaker::setH(const float & x)
{ 
	m_grid.clear();
	m_grid.setGridSize(x);
	std::vector<ITetrahedron *>::iterator it = m_tets.begin();
	for(;it!=m_tets.end();++it) {
		delete *it;
	}
	m_tets.clear();
}

void GridMaker::addCell(const aphid::Vector3F & p)
{
	const sdb::Coord3 c = m_grid.gridCoord((const float *)&p );
	if(m_grid.findCell(c) ) 
		return;		
/// red only
	BccNode * node15 = new BccNode;
	node15->pos = m_grid.coordToCellCenter(c);
	node15->prop = -4;
	node15->key = 15;
	m_grid.insert(c, node15 );
}

void GridMaker::buildGrid()
{
	m_grid.calculateBBox();
	std::cout<<"\n grid n cell "<<m_grid.size()
			<<"\n bbx "<<m_grid.boundingBox();
	
	m_grid.begin();
	while(!m_grid.end() ) {
		m_grid.addBlueNodes(m_grid.coordToCellCenter(m_grid.key() ) );
		m_grid.next();
	}
}

void GridMaker::buildMesh()
{
	std::vector<ITetrahedron *>::iterator it = m_tets.begin();
	for(;it!=m_tets.end();++it) {
		delete *it;
	}
	m_tets.clear();
	m_grid.buildTetrahedrons(m_tets);
}

void GridMaker::buildGraph()
{
	m_graphEdges.clear();
	int v0, v1, v2, v3;
	const int n = numTetrahedrons();
	int i = 0;
	for(; i<n; ++i) {
		const ITetrahedron * t = m_tets[i];
		v0 = t->iv0;
		v1 = t->iv1;
		v2 = t->iv2;
		v3 = t->iv3;
		m_graphEdges.insert(sdb::Coord2(v0, v1).ordered() );
		m_graphEdges.insert(sdb::Coord2(v0, v2).ordered() );
		m_graphEdges.insert(sdb::Coord2(v0, v3).ordered() );
		m_graphEdges.insert(sdb::Coord2(v1, v2).ordered() );
		m_graphEdges.insert(sdb::Coord2(v1, v3).ordered() );
		m_graphEdges.insert(sdb::Coord2(v2, v3).ordered() );
	}
	
	std::map<int, std::vector<int> > vvemap;
	
	int c = 0;
	m_graphEdges.begin();
	while(!m_graphEdges.end() ) {
	
		int v0 = m_graphEdges.key().x;
		vvemap[v0].push_back(c);
		
		int v1 = m_graphEdges.key().y;
		vvemap[v1].push_back(c);
		
		c++;
		m_graphEdges.next();
	}
	
	m_vvEdgeBegins.clear();
	m_vvEdgeInds.clear();
	
	int nvve = 0;
	std::map<int, std::vector<int> >::iterator it = vvemap.begin();
	for(;it!=vvemap.end();++it) {
		m_vvEdgeBegins.push_back(nvve);
		
		pushIndices(it->second);
		nvve += (it->second).size();
		
		it->second.clear();
	}
	vvemap.clear();
}

void GridMaker::pushIndices(const std::vector<int> & a)
{
	std::vector<int>::const_iterator it = a.begin();
	for(;it!=a.end();++it) 
		m_vvEdgeInds.push_back(*it);
}

int GridMaker::numEdgeIndices() const
{ return m_vvEdgeInds.size(); }

int GridMaker::numNodes()
{ return m_grid.numNodes(); }

int GridMaker::numEdges()
{ return m_graphEdges.size(); }

int GridMaker::numTetrahedrons() const
{ return m_tets.size(); }

BccTetraGrid * GridMaker::grid()
{ return &m_grid; }

}
