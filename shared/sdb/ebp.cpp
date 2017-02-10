/*
 *  ebp.cpp
 *  
 *
 *  Created by jian zhang on 2/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ebp.h"
#include <math/miscfuncs.h>

namespace aphid {

EbpNode::EbpNode()
{}

EbpNode::~EbpNode()
{}

EbpCell::EbpCell(Entity * parent) :
sdb::Array<int, EbpNode >(parent)
{}

EbpCell::~EbpCell()
{}

void EbpCell::clear()
{ 
	TParent::clear();
}

EbpGrid::EbpGrid()
{}

EbpGrid::~EbpGrid()
{}

void EbpGrid::fillBox(const BoundingBox & b,
				const float & h)
{
	clear();
	setLevel0CellSize(h);
	
	const int s = level0CoordStride();
	const sdb::Coord4 lc = cellCoordAtLevel(b.getMin(), 0);
	const sdb::Coord4 hc = cellCoordAtLevel(b.getMax(), 0);
	const int dimx = (hc.x - lc.x) / s + 1;
	const int dimy = (hc.y - lc.y) / s + 1;
	const int dimz = (hc.z - lc.z) / s + 1;
	const float fh = finestCellSize();
	
	const Vector3F ori(fh * (lc.x + s/2),
						fh * (lc.y + s/2),
						fh * (lc.z + s/2));
						
	int i, j, k;
	sdb::Coord4 sc;
	sc.w = 0;
	for(k=0; k<dimz;++k) {
		sc.z = lc.z + s * k;
			for(j=0; j<dimy;++j) {
			sc.y = lc.y + s * j;
			for(i=0; i<dimx;++i) {
				sc.x = lc.x + s * i;
				EbpCell * cell = findCell(sc);
				if(!cell) { 
					addCell(sc);
				}
				
			}
		}
	}
	
	calculateBBox();
}

void EbpGrid::insertNodeAtLevel(int level)
{
	const float hgz = levelCellSize(level) * .493f;
	begin();
	while(!end() ) {
		if(key().w == level) {
			EbpNode * par = new EbpNode;
			
			Vector3F r(RandomFn11(), RandomFn11(), RandomFn11() );
			r.normalize();
			
			par->pos = cellCenter(key() ) + r * (hgz * RandomF01()) ;
			par->index = -1;
			value()->insert(0, par);
			
		}
		next();
	}
}

void EbpGrid::extractPos(Vector3F * dst)
{
	begin();
	while(!end() ) {
		
		EbpCell * cell = value();
		cell->begin();
		while(!cell->end() ) {
			
			const EbpNode * node = cell->value();
			dst[node->index] = node->pos;
			cell->next();
		}
		
		next();
	}
}

void EbpGrid::cachePositions()
{
	const int n = countNodes();
	m_pos.reset(new Vector3F[n]);
	extractPos(m_pos.get());
}

const Vector3F * EbpGrid::positions() const
{ return m_pos.get(); }

const int & EbpGrid::numParticles() const
{ return numNodes(); }

void EbpGrid::update()
{
	Vector3F frepel;
	begin();
	while(!end() ) {
		
		EbpCell * cell = value();
		cell->begin();
		while(!cell->end() ) {
			
			EbpNode * node = cell->value();
			
			frepel.set(0.f,0.f,0.f);
			repelForce(frepel, cell, node);
			repelForceInCell(frepel, cell, node);
			
			node->pos += frepel;
			
			cell->next();
		}
		
		next();
	}
	
	extractPos(m_pos.get());
	
	std::cout.flush();
}

void EbpGrid::repelForce(Vector3F & frepel,
						EbpCell * cell,
						const EbpNode * node)
{
	int i=0;
	for(;i<cell->numNeighbors();++i) {
		repelForceInCell(frepel, 
					static_cast<EbpCell *>(cell->neighbor(i) ), node);
	}
}

void EbpGrid::repelForceInCell(Vector3F & frepel,
					EbpCell * cell,
					const EbpNode * node)
{
	if(!cell) return;
	
	Vector3F vd;
	float l;
	cell->begin();
	while(!cell->end() ) {
		
		const EbpNode * nei = cell->value();
		
		if(nei->index != node->index) {
			vd = node->pos - m_pos.get()[nei->index];
			l = vd.length();
			vd /= l;
			l *= m_repelDistance;
			frepel += vd * std::exp(-8.f*l*l);
		}
		
		cell->next();
	}
}

void EbpGrid::clear()
{
	TParent::clear(); 
}


}
