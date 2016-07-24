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

void AdaptiveBccMesher::enforceBoundary(std::vector<sdb::Coord4 > & ks)
{
	while(ks.size() > 0) {
/// first one
		const sdb::Coord4 c = ks[0];
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
/// rm first one
		ks.erase(ks.begin() );
	}
}

void AdaptiveBccMesher::subdivideGridByBox(const aphid::BoundingBox & bx,
							const int & level,
							std::vector<aphid::sdb::Coord4 > & divided)
{
	sdb::Coord4 lc = m_grid.cellCoordAtLevel(bx.getMin(), level);
	sdb::Coord4 hc = m_grid.cellCoordAtLevel(bx.getMax(), level);
	const int s = m_grid.levelCoordStride(level);
	//std::cout<<"\n dirty box"<<lc<<" "<<hc;
	int dimx = (hc.x - lc.x) / s; if(dimx ==0) dimx=1;
	int dimy = (hc.y - lc.y) / s; if(dimy ==0) dimy=1;
	int dimz = (hc.z - lc.z) / s; if(dimz ==0) dimz=1;
	//std::cout<<"\n grid dim "<<dimx<<" x "<<dimy<<" x "<<dimz;
	for(int k=0; k<dimz;++k) {
		for(int j=0; j<dimy;++j) {
			for(int i=0; i<dimx;++i) {
				sdb::Coord4 c(lc.x + s * i,
								lc.y + s * j,
								lc.z + s * k,
								level);
				if(m_grid.subdivideCell(c ))
					divided.push_back(c);
				
			}
		}
	}
}

void AdaptiveBccMesher::subdivideGridByPnt(const aphid::Vector3F & pref,
							const int & level,
							std::vector<aphid::sdb::Coord4 > & divided)
{
	sdb::Coord4 c = m_grid.cellCoordAtLevel(pref, level);
	if(m_grid.subdivideCell(c )) {
		divided.push_back(c); 
		std::cout<<"\n split cell l"<<c.w;
	}

}

}
