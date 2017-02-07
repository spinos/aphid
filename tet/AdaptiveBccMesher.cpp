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

namespace aphid {

namespace ttg {
AdaptiveBccMesher::AdaptiveBccMesher()
{}

AdaptiveBccMesher::~AdaptiveBccMesher()
{ internalClear(); }

void AdaptiveBccMesher::internalClear()
{
	m_grid.clear();
	m_grid.resetNumNodes();
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

void AdaptiveBccMesher::fillBox(const aphid::BoundingBox & b,
				const float & h)
{
	m_grid.clear();
	m_grid.resetNumNodes();
	m_grid.setLevel0CellSize(h);
	m_grid.subdivideToLevel(b, 0);
	m_grid.calculateBBox();
	std::cout<<"\n adaptive bcc mesher grid bbx "<<m_grid.boundingBox();
}

void AdaptiveBccMesher::buildMesh()
{ 
	clearTetra();
	m_grid.begin();
	while(!m_grid.end() ) {
/// undivided
		if(!m_grid.value()->hasChild() ) 
			m_grid.value()->connectTetrahedrons(m_tets, m_grid.key(), &m_grid);
			
		m_grid.next();
	}
}

const int & AdaptiveBccMesher::numVertices() const
{ return m_grid.numNodes(); }

int AdaptiveBccMesher::numTetrahedrons() const
{ return m_tets.size(); }

AdaptiveBccGrid3 * AdaptiveBccMesher::grid()
{ return &m_grid; }

const ITetrahedron * AdaptiveBccMesher::tetra(const int & i) const
{ return m_tets[i]; }

const std::vector<ITetrahedron *> & AdaptiveBccMesher::tetrahedrons() const
{ return m_tets; }

void AdaptiveBccMesher::enforceBoundary(std::vector<sdb::Coord4 > & ks)
{
	while(ks.size() > 0) {
/// first one
		const sdb::Coord4 c = ks[0];
		
/// per face
		for(int i=0; i< 6;++i) {
			const sdb::Coord4 nei = m_grid.neighborCoord(c, i);
			const sdb::Coord4 par = m_grid.parentCoord(nei);
			if(m_grid.findCell(par) ) {
				if(!m_grid.findCell(nei) ) {
					if(m_grid.subdivideCell(par ) )
/// last one
						ks.push_back(par);
				}
			}
		}
		
/// per edge
		for(int i=14; i< 26;++i) {
			const sdb::Coord4 nei = m_grid.neighborCoord(c, i);
			const sdb::Coord4 par = m_grid.parentCoord(nei);
			if(m_grid.findCell(par) ) {
				if(!m_grid.findCell(nei) ) {
					if(m_grid.subdivideCell(par ) )
/// last one
						ks.push_back(par);
				}
			}
		}
		
/// rm first one
		ks.erase(ks.begin() );
	}
}

}
}
