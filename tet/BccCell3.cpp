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
{ 
	m_hasChild = false; 
	m_parentCell = NULL;
}

BccCell3::~BccCell3()
{}

void BccCell3::setHasChild()
{ m_hasChild = true; }

void BccCell3::setParentCell(BccCell3 * x, const int & i)
{ 
	m_parentCell = x;
	m_childI = i;
}

void BccCell3::insertRed(const Vector3F & pref)
{
	BccNode * node15 = new BccNode;
	node15->pos = pref;
	node15->prop = BccCell::NRed;
	node15->key = 15;
	insert(15, node15 );
}

BccNode * BccCell3::findBlue(const Vector3F & pref)
{
	const Vector3F & center = find(15)->pos;
	const int k = sdb::gdt::KeyToBlue(pref, center);
	return find(k);
}

void BccCell3::insertBlue(const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
	if(cellCoord.w > 0) 
		return;
		
	const BccNode * redN = find(15);
	const Vector3F & redP = redN->pos;
	const float & gz = grid->levelCellSize(cellCoord.w);
	int i;
	
	Vector3F q;
	for(i=0;i<8;++i) {
		
		if(blueNode(i, cellCoord, grid) )
			continue;
		
		sdb::gdt::GetVertexNodeOffset(q, i);
		q = redP + q * gz * .5f;
			
		BccNode * ni = new BccNode;
		ni->key = i + 6;
		ni->prop = BccCell::NBlue;
		ni->pos = q;
		insert(i + 6, ni);
		
	}
}

void BccCell3::insertFaceOnBoundary(const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
	const float & gz = grid->levelCellSize(cellCoord.w);
	const BccNode * redN = find(15);
	const Vector3F & redP = redN->pos;
	
	for(int i=0; i<6;++i) {
		if(!grid->findNeighborCell(cellCoord, i ) ) {
			BccNode * ni = new BccNode;
			ni->key = i;
			ni->prop = BccCell::NFace;
			
			sdb::gdt::GetFaceNodeOffset(ni->pos, i);
			ni->pos = redP + ni->pos * .5f * gz;
			insert(i, ni);
		}
	}
}

void BccCell3::insertYellow(const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
	const BccNode * redN = find(15);
	const Vector3F & redP = redN->pos;
	for(int i=0; i<6;++i) {
		if(find(i) )
			continue;
			
		if(yellowNode(i, cellCoord, grid) )
			continue;
			
		BccCell3 * nei = grid->findNeighborCell(cellCoord, i);
		BccNode * ni = new BccNode;
		ni->key = 15000 + i;
		ni->prop = BccCell::NYellow;
		ni->pos = (redP + nei->find(15)->pos ) * .5f;
		insert(15000 + i, ni);
	}
}

void BccCell3::insertCyan(const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
	for(int i=0;i<12;++i) {
		if(cyanNode(i, cellCoord, grid) )
			continue;
			
		BccNode * ni = new BccNode;
		ni->key = sdb::gdt::TwelveBlueBlueEdges[i][2];
		ni->prop = BccCell::NCyan;
		
		BccNode * b1 = blueNode(sdb::gdt::TwelveBlueBlueEdges[i][0] - 6,
								cellCoord, grid);
		BccNode * b2 = blueNode(sdb::gdt::TwelveBlueBlueEdges[i][1] - 6,
								cellCoord, grid);
		ni->pos = (b1->pos + b2->pos ) * .5f;
		
		insert(ni->key, ni);
	}
}

BccNode * BccCell3::blueNode(const int & i,
					const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
/// level > 0 blue derived from parent red blue yellow cyan
	if(m_parentCell)
		return derivedBlueNode(i, cellCoord, grid);
		
	BccNode * node = find(i+6);
	if(!node) 
		node = findBlueNodeInNeighbor(i, cellCoord, grid);
	return node;
}

BccNode * BccCell3::yellowNode(const int & i,
					const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
/// face node as yellow
	BccNode * node = find(i);
	if(node)
		return node;
		
	node = find(15000 + i);
	if(!node) {
		BccCell3 * nei = grid->findNeighborCell(cellCoord, i);
		node = nei->find(15000 + sdb::gdt::SixNeighborOnFace[i][3]);
	}
	return node;
}

BccNode * BccCell3::cyanNode(const int & i,
					const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
	BccNode * node = find(sdb::gdt::TwelveBlueBlueEdges[i][2]);
	if(!node)
		node = findCyanNodeInNeighbor(i, cellCoord, grid);
		
	return node;
}

BccNode * BccCell3::findCyanNodeInNeighbor(const int & i,
					const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
	for(int j = 0;j<3;++j) {
		BccCell3 * cell = grid->findEdgeNeighborCell(cellCoord, i, j);
		if(cell) {
			BccNode * node = cell->find(sdb::gdt::ThreeNeighborOnEdge[i*3+j][3]);
			if(node)
				return node;
		}
	}
	return NULL;
}

BccNode * BccCell3::findBlueNodeInNeighbor(const int & i,
					const sdb::Coord4 & cellCoord,
					sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
	const float & gz = grid->levelCellSize(cellCoord.w);
	
	Vector3F q;
	sdb::gdt::GetVertexNodeOffset(q, i);
	q = find(15)->pos + q * gz * .5f;

	int j;
	for(j=0;j<7;++j) {
		int neighborJ = sdb::gdt::GetVertexNeighborJ(i, j);
		
		BccCell3 * neighborCell = grid->findNeighborCell(cellCoord, neighborJ);
		if(neighborCell) {
			BccNode * node = neighborCell->findBlue(q);
			if(node)
				return node;
		}
	}
	return NULL;
}

BccNode * BccCell3::derivedBlueNode(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					aphid::sdb::AdaptiveGrid3<BccCell3, BccNode > * grid)
{
	sdb::Coord4 pc = grid->parentCoord(cellCoord);
	if(sdb::gdt::isEighChildBlueInParentIsBlue(m_childI, i) )
		return m_parentCell->blueNode(sdb::gdt::EightChildBlueInParentInd[m_childI][i],
										pc, grid);
		
	if(sdb::gdt::isEighChildBlueInParentIsCyan(m_childI, i) )
		return m_parentCell->cyanNode(sdb::gdt::EightChildBlueInParentInd[m_childI][i],
										pc, grid);
	
	if(sdb::gdt::isEighChildBlueInParentIsYellow(m_childI, i) )
		return m_parentCell->yellowNode(sdb::gdt::EightChildBlueInParentInd[m_childI][i],
										pc, grid);
	
	return m_parentCell->find(15);
}

const bool & BccCell3::hasChild() const
{ return m_hasChild; }

}