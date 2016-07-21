/*
 *  BccCell3.cpp
 *  foo
 *
 *  Created by jian zhang on 7/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BccCell3.h"

using namespace aphid;

namespace ttg {

BccCell3::BccCell3(Entity * parent) : sdb::Array<int, BccNode >(parent)
{ m_hasChild = false; }

BccCell3::~BccCell3()
{}

void BccCell3::setHasChild()
{ m_hasChild = true; }

void BccCell3::insertRed(const Vector3F & pref)
{
	BccNode * node15 = new BccNode;
	node15->pos = pref;
	node15->prop = BccCell::NRed;
	node15->key = 15;
	insert(15, node15 );
}

BccNode * BccCell3::findBlue(const aphid::Vector3F & pref)
{
	const Vector3F & center = find(15)->pos;
	const int k = sdb::gdt::KeyToBlue(pref, center);
	return find(k);
}

void BccCell3::insertBlue(const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
	const BccNode * redN = find(15);
	const Vector3F & redP = redN->pos;
	if(cellCoord.w < 1)
		insertBlue0(redP, cellCoord, grid);
	else 
		insertBlue1(redP, cellCoord, grid);
}

void BccCell3::insertBlue0(const Vector3F & center,
					const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
	const float & gz = grid->levelCellSize(cellCoord.w);
	int i, j;
	
	Vector3F q;
	for(i=0;i<8;++i) {
		
		bool toadd = true;
		sdb::gdt::GetVertexNodeOffset(q, i);
		q = center + q * gz * .5f;
		
		for(j=0;j<7;++j) {
			int neighborJ = sdb::gdt::GetVertexNeighborJ(i, j);
			
			BccCell3 * neighborCell = grid->findNeighborCell(cellCoord, neighborJ);
			if(neighborCell) {
				if(neighborCell->findBlue(q) ) {
					toadd = false;
					break;
				}
			}
		}
		
		if(toadd) {
			BccNode * ni = new BccNode;
			ni->key = i + 6;
			ni->prop = BccCell::NBlue;
			ni->pos = q;
			insert(i + 6, ni);
		}
	}
}

void BccCell3::insertBlue1(const Vector3F & center,
				const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{

}

void BccCell3::insertFaceOnBoundary(const aphid::sdb::Coord4 & cellCoord,
					aphid::sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
	const float & gz = grid->levelCellSize(cellCoord.w);
	const BccNode * redN = find(15);
	const Vector3F & redP = redN->pos;
	
	for(int i=0; i<6;++i) {
		if(grid->isCellFaceOnBoundray(cellCoord, i ) ) {
			BccNode * ni = new BccNode;
			ni->key = i;
			ni->prop = BccCell::NFace;
			
			sdb::gdt::GetFaceNodeOffset(ni->pos, i);
			ni->pos = redP + ni->pos * .5f * gz;
			insert(i, ni);
		}
	}
}

const bool & BccCell3::hasChild() const
{ return m_hasChild; }

}