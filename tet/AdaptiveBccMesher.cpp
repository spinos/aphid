/*
 *  AdaptiveBccMesher.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaptiveBccMesher.h"
#include <iostream>

using namespace aphid;

namespace ttg {
AdaptiveBccMesher::AdaptiveBccMesher() :
m_numVert(0)
{}

AdaptiveBccMesher::~AdaptiveBccMesher()
{ internalClear(); }

void AdaptiveBccMesher::internalClear()
{
	m_grid.clear();
	m_triInds.clear();
	clearTetra();
}

void AdaptiveBccMesher::clearTetra()
{
	std::vector<ITetrahedron *>::iterator it = m_tets.begin();
	for(;it!=m_tets.end();++it) {
		delete *it;
	}
	m_tets.clear();
}

void AdaptiveBccMesher::setH(const float & x)
{ 
	m_grid.clear();
	m_grid.setLevel0CellSize(x);
	std::cout<<"\n grid max level "<<m_grid.maxLevel();
}

void AdaptiveBccMesher::addCell(const aphid::Vector3F & p)
{ m_grid.addCell(p ); }

void AdaptiveBccMesher::buildGrid()
{
	m_grid.calculateBBox();
	std::cout<<"\n grid n cell "<<m_grid.size()
			<<"\n bbx "<<m_grid.boundingBox();
	
	m_grid.build();
}

void AdaptiveBccMesher::buildMesh()
{ 
	clearTetra();
	m_numVert = m_grid.countNodes();
	m_grid.begin();
	while(!m_grid.end() ) {
/// undivided
		if(!m_grid.value()->hasChild() ) 
			m_grid.value()->connectTetrahedrons(m_tets, m_grid.key(), &m_grid);
			
		m_grid.next();
	}
}

const int & AdaptiveBccMesher::numVertices() const
{ return m_numVert; }

int AdaptiveBccMesher::numTetrahedrons() const
{ return m_tets.size(); }

AdaptiveBccGrid3 * AdaptiveBccMesher::grid()
{ return &m_grid; }

const ITetrahedron * AdaptiveBccMesher::tetra(const int & i) const
{ return m_tets[i]; }

int AdaptiveBccMesher::numTriangles() const
{ return m_triInds.size(); }

const sdb::Coord3 & AdaptiveBccMesher::triangleInd(const int & i) const
{ return m_triInds[i]; }

void AdaptiveBccMesher::buildMesh1()
{
	//m_grid.cutEdges();
	clearTetra();
	m_grid.countNodes();
	RedBlueRefine rbr;
	aphid::sdb::Array<aphid::sdb::Coord3, IFace > tris;
	m_grid.begin();
	while(!m_grid.end() ) {
	//	BccCell fCell(m_grid.coordToCellCenter(m_grid.key() ) );
		
	//	fCell.connectRefinedTetrahedrons(m_tets, tris, rbr, m_grid.value(), &m_grid, m_grid.key() );
		m_grid.next();
	}
	
	m_triInds.clear();
	
	tris.begin();
	while(!tris.end() ) {
	
		m_triInds.push_back(tris.key() );
		tris.next();
	}
	
	tris.clear();
}

void AdaptiveBccMesher::enforceBoundary(const std::vector<sdb::Coord4 > & ks)
{
	std::vector<sdb::Coord4 >::const_iterator it = ks.begin();
	for(;it!=ks.end();++it) {
		
		for(int i=0; i< 6;++i) {
			const sdb::Coord4 nei = m_grid.neighborCoord(*it, i);

			if(!m_grid.findCell(nei) ) {
				const sdb::Coord4 par = m_grid.parentCoord(nei);
				m_grid.subdivideCell(par );
			}
		}
	}
}

}
