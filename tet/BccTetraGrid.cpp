/*
 *  BccTetraGrid.cpp
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <BccTetraGrid.h>

using namespace aphid;

namespace ttg {

float BccCell::TwentySixNeighborOffset[26][3] = {
{-1.f, 0.f, 0.f}, // face
{ 1.f, 0.f, 0.f},
{ 0.f,-1.f, 0.f},
{ 0.f, 1.f, 0.f},
{ 0.f, 0.f,-1.f},
{ 0.f, 0.f, 1.f},
{-1.f,-1.f,-1.f}, // vertex
{ 1.f,-1.f,-1.f},
{-1.f, 1.f,-1.f},
{ 1.f, 1.f,-1.f},
{-1.f,-1.f, 1.f},
{ 1.f,-1.f, 1.f},
{-1.f, 1.f, 1.f},
{ 1.f, 1.f, 1.f},
{-1.f, 0.f,-1.f}, // edge
{ 1.f, 0.f,-1.f},
{-1.f, 0.f, 1.f},
{ 1.f, 0.f, 1.f},
{ 0.f,-1.f,-1.f},
{ 0.f, 1.f,-1.f},
{ 0.f,-1.f, 1.f},
{ 0.f, 1.f, 1.f},
{-1.f,-1.f, 0.f},
{ 1.f,-1.f, 0.f},
{-1.f, 1.f, 0.f},
{ 1.f, 1.f, 0.f}
};

/// 3 face, 1 vertex, 3 edge
int BccCell::SevenNeighborOnCorner[8][7] = {
{0, 2, 4,  6, 14, 18, 22},	
{1, 2, 4,  7, 15, 18, 23},
{0, 3, 4,  8, 14, 19, 24},
{1, 3, 4,  9, 15, 19, 25},
{0, 2, 5, 10, 16, 20, 22},
{1, 2, 5, 11, 17, 20, 23},
{0, 3, 5, 12, 16, 21, 24},
{1, 3, 5, 13, 17, 21, 25}
};

BccCell::BccCell(const Vector3F &center )
{ m_center = center; }

const Vector3F * BccCell::centerP() const
{ return &m_center; }

void BccCell::addNodes(sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
						const sdb::Coord3 & cellCoord ) const
{
	const float gsize = grid->gridSize();
	
	Vector3F samp, offset;

/// face 1 - 6 when on border
	int j, i = 0;
	for(;i<6;++i) {
		offset.set(TwentySixNeighborOffset[i][0],
					TwentySixNeighborOffset[i][1],
					TwentySixNeighborOffset[i][2]);
		samp = m_center + offset * gsize;
			
		if(!grid->findCell(samp) ) {
			BccNode * ni = new BccNode;
			ni->key = i;
			
			grid->insert(cellCoord, ni);
		}
	}
	
/// corner 7 - 14 when on border or neighbor has no opposing node
	i=6;
	for(;i<14;++i) {
		
		bool toadd = true;
		
		offset.set(TwentySixNeighborOffset[i][0],
					TwentySixNeighborOffset[i][1],
					TwentySixNeighborOffset[i][2]);
		samp = m_center + offset * gsize * .5f;
		
		for(j=0;j<7;++j) {
			int neighborJ = SevenNeighborOnCorner[i-6][j];
			offset.set(TwentySixNeighborOffset[neighborJ][0],
					TwentySixNeighborOffset[neighborJ][1],
					TwentySixNeighborOffset[neighborJ][2]);
			Vector3F neighborCenter = m_center + offset * gsize;
			if(findNeighborCorner(grid, neighborCenter,
					cornerI(samp, neighborCenter) ) ) {
				toadd = false;
				break;
			}
		}
		
		if(toadd) {
			BccNode * ni = new BccNode;
			ni->key = i;
				
			grid->insert(cellCoord, ni);
		}
	}
}

bool BccCell::findNeighborCorner(sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const Vector3F & pos, int icorner) const
{
	sdb::Array<int, BccNode> * neicell = grid->findCell(pos);
	if(!neicell) 
		return false;
	
	return neicell->find(icorner );
}

int BccCell::cornerI(const Vector3F & corner,
				const Vector3F & center) const
{
	float dx = corner.x - center.x;
	float dy = corner.y - center.y;
	float dz = corner.z - center.z;
	if(dz < 0.f) {
		if(dy < 0.f) {
			if(dx < 0.f) 
				return 6;
			else 
				return 7;
		}
		else {
			if(dx < 0.f) 
				return 8;
			else 
				return 9;
		}
	}
	else {
		if(dy < 0.f) {
			if(dx < 0.f) 
				return 10;
			else 
				return 11;
		}
		else {
			if(dx < 0.f) 
				return 12;
			else 
				return 13;
		}
	}
	return 13;
}

BccTetraGrid::BccTetraGrid() {}
BccTetraGrid::~BccTetraGrid() {}

void BccTetraGrid::buildTetrahedrons()
{
	std::vector<BccCell> cells;
	begin();
	while(!end() ) {
		cells.push_back(BccCell(coordToCellCenter(key() ) ) );
		next();
	}
	
	const int n = cells.size();
	int i=0;
	for(;i<n;++i) {
		const BccCell & c = cells[i];
		c.addNodes(this, gridCoord((const float *)c.centerP() ) );
	}
	cells.clear();
	countNodes();
}

void BccTetraGrid::countNodes()
{
	int c = 0;
	begin();
	while(!end() ) {
		countNodesIn(value(), c );
		next();
	}
}

void BccTetraGrid::countNodesIn(aphid::sdb::Array<int, BccNode> * cell, int & c)
{
	cell->begin();
	while(!cell->end() ) {
		cell->value()->index = c;
		c++;
		cell->next();
	}
}

int BccTetraGrid::numNodes()
{
	int c = 0;
	begin();
	while(!end() ) {
		c+= value()->size();
		next();
	}
	return c;
}

}
