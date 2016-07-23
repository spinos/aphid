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
AdaptiveBccMesher::AdaptiveBccMesher()
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

void AdaptiveBccMesher::fillBox(const aphid::BoundingBox & b,
				const float & h)
{
	m_grid.clear();
	m_grid.setLevel0CellSize(h);
	
	int i, j, k;
	int s = m_grid.level0CoordStride();
	
	sdb::Coord4 lc = m_grid.cellCoordAtLevel(b.getMin(), 0);
	sdb::Coord4 hc = m_grid.cellCoordAtLevel(b.getMax(), 0);
	int dimx = (hc.x - lc.x) / s + 1;
	int dimy = (hc.y - lc.y) / s + 1; 
	int dimz = (hc.z - lc.z) / s + 1;
	float fh = m_grid.finestCellSize();
	std::cout<<"\n level0 cell size "<<h
		<<"\n grid dim "<<dimx<<" x "<<dimy<<" x "<<dimz;
	
	Vector3F ori(fh * (lc.x + s/2),
				fh * (lc.y + s/2),
				fh * (lc.z + s/2));
	
	for(k=0; k<dimz;++k) {
		for(j=0; j<dimy;++j) {
			for(i=0; i<dimx;++i) {
				addCell(ori + Vector3F(i, j, k) * (fh * s) );
			}
		}
	}
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
	m_numVert = 0;
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
			const sdb::Coord4 par = m_grid.parentCoord(nei);
			if(m_grid.findCell(par) ) {
				if(!m_grid.findCell(nei) )
					m_grid.subdivideCell(par );
			}
		}
	}
}

}
