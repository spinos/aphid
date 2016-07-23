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
					AdaptiveGridT * grid)
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
					AdaptiveGridT * grid)
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
					AdaptiveGridT * grid)
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
					AdaptiveGridT * grid)
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
					AdaptiveGridT * grid)
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
					AdaptiveGridT * grid)
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
					AdaptiveGridT * grid)
{
	BccNode * node = find(sdb::gdt::TwelveBlueBlueEdges[i][2]);
	if(!node)
		node = findCyanNodeInNeighbor(i, cellCoord, grid);
		
	return node;
}

BccNode * BccCell3::findCyanNodeInNeighbor(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
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
					AdaptiveGridT * grid)
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
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
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

BccNode * BccCell3::faceNode(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	BccNode * node = yellowNode(i, cellCoord, grid);
	if(node)
		return node;
		
	BccCell3 * nei = grid->findNeighborCell(cellCoord, i);
	if(!nei) {
		std::cout<<"\n [ERROR] no neighbor"<<i<<" in cell"<<cellCoord;
		return NULL;
	}
	node = nei->find(15);

	return node;
}

const bool & BccCell3::hasChild() const
{ return m_hasChild; }

void BccCell3::connectTetrahedrons(std::vector<ITetrahedron *> & dest,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	BccNode * redN = find(15);
	int i = 0, j, k;
/// per face
	for(;i<6;++i) {
		BccNode * faN = faceNode(i, cellCoord, grid);
		if((i&1) == 0) {
			if(faN->key == 15)
				continue;
		}
/// per edge
		for(j=0;j<4;++j) {
			BccNode * cN = blueNode(sdb::gdt::TwentyFourFVBlueBlueEdge[i*4+j][0] - 6,
									cellCoord, grid);
			BccNode * dN = blueNode(sdb::gdt::TwentyFourFVBlueBlueEdge[i*4+j][1] - 6,
									cellCoord, grid);
			
			k = sdb::gdt::TwentyFourFVBlueBlueEdge[i*4+j][2];
			
			if(isFaceDivided(i, cellCoord, grid) )
				addFourTetra(dest, i, j, k, cellCoord, grid, redN, faN, cN, dN);
			else if(isEdgeDivided(k, cellCoord, grid) )
				addTwoTetra(dest, k, cellCoord, grid, redN, faN, cN, dN);
			else
				addOneTetra(dest, redN, faN, cN, dN);
		}
	}
}

void BccCell3::addOneTetra(std::vector<ITetrahedron *> & dest,
					BccNode * A, BccNode * B, BccNode * C, BccNode * D)
{
	ITetrahedron * t = new ITetrahedron;
	resetTetrahedronNeighbors(*t);
	setTetrahedronVertices(*t, A->index, B->index, C->index, D->index);
	t->index = dest.size();
	dest.push_back(t);
}

void BccCell3::addTwoTetra(std::vector<ITetrahedron *> & dest,
					const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid,
					BccNode * A, BccNode * B, BccNode * C, BccNode * D)
{
	BccNode * cyanN = cyanNode(i, cellCoord, grid);
	addOneTetra(dest, A, B, cyanN, D);
	addOneTetra(dest, A, B, C, cyanN);
}

void BccCell3::addFourTetra(std::vector<ITetrahedron *> & dest,
					const int & i,
					const int & j,
					const int & k,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid,
					BccNode * A, BccNode * B, BccNode * C, BccNode * D)
{
	BccNode * cyanN = cyanNode(k, cellCoord, grid);
	
	sdb::Coord4 nc = grid->neighborCoord(cellCoord, i);
	
	sdb::Coord4 ncc0 = grid->childCoord(nc, sdb::gdt::TwentyFourFVEdgeNeighborChildFace[i*4+j][0]);
	BccCell3 * cell0 = grid->findCell(ncc0);
	if(!cell0) {
		std::cout<<"\n [ERROR] no neighbor child "<<ncc0<<" in cell"<<cellCoord;
		return;
	}
	
	sdb::Coord4 ncc1 = grid->childCoord(nc, sdb::gdt::TwentyFourFVEdgeNeighborChildFace[i*4+j][1]);
	BccCell3 * cell1 = grid->findCell(ncc1);
	if(!cell1) {
		std::cout<<"\n [ERROR] no neighbor child "<<ncc1<<" in cell"<<cellCoord;
		return;
	}
	
	BccNode * f0 = cell0->faceNode(sdb::gdt::TwentyFourFVEdgeNeighborChildFace[i*4+j][2], 
						ncc0, grid);
	if(!f0) {
		std::cout<<"\n [ERROR] no neighbor child face "<<ncc0;
		return;
	}
						
	BccNode * f1 = cell1->faceNode(sdb::gdt::TwentyFourFVEdgeNeighborChildFace[i*4+j][2], 
						ncc1, grid);	
	if(!f1) {
		std::cout<<"\n [ERROR] no neighbor child face "<<ncc1;
		return;
	}
	
	addOneTetra(dest, A, f0, C, cyanN);
	addOneTetra(dest, A, B, f0, cyanN);
	addOneTetra(dest, A, B, cyanN, f1);
	addOneTetra(dest, A, f1, cyanN, D);
}

bool BccCell3::isFaceDivided(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{ 
	BccCell3 * nei = grid->findNeighborCell(cellCoord, i);
	if(!nei) 
		return false;

	return nei->hasChild();
}

bool BccCell3::isEdgeDivided(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	for(int j = 0;j<3;++j) {
		BccCell3 * cell = grid->findEdgeNeighborCell(cellCoord, i, j);
		if(cell) {
			if(cell->hasChild())
				return true;
		}
	}
	return false; 
}

}