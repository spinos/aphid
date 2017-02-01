/*
 *  BccCell3.cpp
 *  ttg
 *
 *  Created by jian zhang on 7/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BccCell3.h"

namespace aphid {

namespace ttg {

BccCell3::BccCell3(Entity * parent) : sdb::Array<int, BccNode3 >(parent)
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
	BccNode3 * node15 = new BccNode3;
	node15->val = 1e9f;
	node15->pos = pref;
	node15->prop = sdb::gdt::NRed;
	node15->key = 15;
	node15->index = -1;
	insert(15, node15 );
}

BccNode3 * BccCell3::findBlue(const Vector3F & pref)
{
	const Vector3F & center = find(15)->pos;
	const int k = sdb::gdt::KeyToBlue(pref, center);
	return find(k);
}

void BccCell3::insertBlue(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	BccNode3 * redN = find(15);
	if(cellCoord.w > 0) {
		//findRedValueFrontBlue(redN, cellCoord, grid, fld);
		return;
	}
		
	const Vector3F & redP = redN->pos;
	const float & gz = grid->levelCellSize(cellCoord.w);
	int i;
	
	Vector3F q;
	for(i=0;i<8;++i) {
		
		if(blueNode(i, cellCoord, grid) )
			continue;
		
		sdb::gdt::GetVertexNodeOffset(q, i);
		q = redP + q * gz * .5f;
			
		BccNode3 * ni = new BccNode3;
		ni->val = 1e9f;
		ni->key = i + 6;
		ni->prop = sdb::gdt::NBlue;
		ni->pos = q;
		ni->index = -1;
		insert(i + 6, ni);
		
	}

}

void BccCell3::insertFaceOnBoundary(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	const float & gz = grid->levelCellSize(cellCoord.w);
	const BccNode3 * redN = find(15);
	const Vector3F & redP = redN->pos;
	
	for(int i=0; i<6;++i) {
		if(find(i) ) continue;
		
		if(!grid->findNeighborCell(cellCoord, i ) ) {
			BccNode3 * ni = new BccNode3;
			ni->val = 1e9f;
			ni->key = i;
			ni->prop = sdb::gdt::NFace;
			ni->index = -1;
			sdb::gdt::GetFaceNodeOffset(ni->pos, i);
			ni->pos = redP + ni->pos * .5f * gz;
			insert(i, ni);
		}
	}
}

void BccCell3::insertYellow(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	const BccNode3 * redN = find(15);
	const Vector3F & redP = redN->pos;
	for(int i=0; i<6;++i) {
		if(find(i) )
			continue;
			
		if(yellowNode(i, cellCoord, grid) )
			continue;
			
		BccCell3 * nei = grid->findNeighborCell(cellCoord, i);
		BccNode3 * ni = new BccNode3;
		ni->key = 15000 + i;
		ni->prop = sdb::gdt::NYellow;
		const BccNode3 * neiRedN = nei->find(15);
		ni->pos = (redP + neiRedN->pos ) * .5f;
		//const IDistanceEdge * eg = fld->edge(redN->index, neiRedN->index);
		//if(eg)
		//	ni->val = eg->val;
		//else
			ni->val = 1e9f;
		ni->index = -1;
		insert(15000 + i, ni);
	}
}

void BccCell3::insertCyan(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	for(int i=0;i<12;++i) {
		if(cyanNode(i, cellCoord, grid) )
			continue;
			
		BccNode3 * ni = new BccNode3;
		ni->key = sdb::gdt::TwelveBlueBlueEdges[i][2];
		ni->prop = sdb::gdt::NCyan;
		
		BccNode3 * b1 = blueNode(sdb::gdt::TwelveBlueBlueEdges[i][0] - 6,
								cellCoord, grid);
		BccNode3 * b2 = blueNode(sdb::gdt::TwelveBlueBlueEdges[i][1] - 6,
								cellCoord, grid);
		ni->pos = (b1->pos + b2->pos ) * .5f;
		//const IDistanceEdge * eg = fld->edge(b1->index, b2->index);
		//if(eg)
		//	ni->val = eg->val;
		//else
			ni->val = 1e9f;
		ni->index = -1;
		insert(ni->key, ni);
	}
}

BccNode3 * BccCell3::blueNode(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
/// level > 0 blue derived from parent red blue yellow cyan
	if(m_parentCell)
		return derivedBlueNode(i, cellCoord, grid);
		
	BccNode3 * node = find(i+6);
	if(!node) 
		node = findBlueNodeInNeighbor(i, cellCoord, grid);
	return node;
}

BccNode3 * BccCell3::yellowNode(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
/// face node as yellow
	BccNode3 * node = find(i);
	if(node)
		return node;
		
	node = find(15000 + i);
	if(node)
		return node;
		
	BccCell3 * nei = grid->findNeighborCell(cellCoord, i);
	if(!nei)
		return NULL;
		
/// opposite face in neighbor
	node = nei->find(sdb::gdt::SixNeighborOnFace[i][3]);
	if(node)
		return node;
		
	return nei->find(15000 + sdb::gdt::SixNeighborOnFace[i][3]);
}

BccNode3 * BccCell3::cyanNode(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	BccNode3 * node = find(sdb::gdt::TwelveBlueBlueEdges[i][2]);
	if(!node)
		node = findCyanNodeInNeighbor(i, cellCoord, grid);
		
	return node;
}

BccNode3 * BccCell3::findCyanNodeInNeighbor(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	for(int j = 0;j<3;++j) {
		BccCell3 * cell = grid->findEdgeNeighborCell(cellCoord, i, j);
		if(cell) {
			BccNode3 * node = cell->find(sdb::gdt::ThreeNeighborOnEdge[i*3+j][3]);
			if(node)
				return node;
		}
	}
	return NULL;
}

BccNode3 * BccCell3::findBlueNodeInNeighbor(const int & i,
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
			BccNode3 * node = neighborCell->findBlue(q);
			if(node)
				return node;
		}
	}
	return NULL;
}

BccNode3 * BccCell3::derivedBlueNode(const int & i,
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

BccNode3 * BccCell3::faceNode(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	BccNode3 * node = yellowNode(i, cellCoord, grid);
	if(node)
		return node;
		
	BccCell3 * nei = grid->findNeighborCell(cellCoord, i);
	if(!nei) {
		std::cout<<"\n [ERROR] no neighbor"<<i<<" in cell"<<cellCoord;
		return NULL;
	}
	
	return nei->find(15);
}

const bool & BccCell3::hasChild() const
{ return m_hasChild; }

void BccCell3::connectTetrahedrons(std::vector<ITetrahedron *> & dest,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	BccNode3 * redN = find(15);
	int i = 0, j, k;
/// per face
	for(;i<6;++i) {
		BccNode3 * faN = faceNode(i, cellCoord, grid);
		
/// per edge
		for(j=0;j<4;++j) {
			BccNode3 * cN = blueNode(sdb::gdt::TwentyFourFVBlueBlueEdge[i*4+j][0] - 6,
									cellCoord, grid);
			BccNode3 * dN = blueNode(sdb::gdt::TwentyFourFVBlueBlueEdge[i*4+j][1] - 6,
									cellCoord, grid);
			
			k = sdb::gdt::TwentyFourFVBlueBlueEdge[i*4+j][2];
			
			if(isFaceDivided(i, cellCoord, grid) )
				addFourTetra(dest, i, j, k, cellCoord, grid, redN, faN, cN, dN);
			else if(isEdgeDivided(k, cellCoord, grid) )
				addTwoTetra(dest, k, cellCoord, grid, redN, faN, cN, dN);
			else {
				if((i&1) == 0) {
					if(faN->key == 15)
						continue;
				}
				addOneTetra(dest, redN, faN, cN, dN);
			}
		}
	}
}

void BccCell3::addOneTetra(std::vector<ITetrahedron *> & dest,
					BccNode3 * A, BccNode3 * B, BccNode3 * C, BccNode3 * D)
{
	ITetrahedron * t = new ITetrahedron;
	resetTetrahedronNeighbors(*t);
	setTetrahedronVertices(*t, A->index, B->index, C->index, D->index);
	t->index = dest.size();
	dest.push_back(t);
}

void BccCell3::addTwoTetra(std::vector<ITetrahedron *> & dest,
					const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid,
					BccNode3 * A, BccNode3 * B, BccNode3 * C, BccNode3 * D)
{
	BccNode3 * cyanN = cyanNode(i, cellCoord, grid);
	addOneTetra(dest, A, B, cyanN, D);
	addOneTetra(dest, A, B, C, cyanN);
}

void BccCell3::addFourTetra(std::vector<ITetrahedron *> & dest,
					const int & i,
					const int & j,
					const int & k,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid,
					BccNode3 * A, BccNode3 * B, BccNode3 * C, BccNode3 * D)
{
	BccNode3 * cyanN = cyanNode(k, cellCoord, grid);
	
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
	
	BccNode3 * f0 = cell0->faceNode(sdb::gdt::TwentyFourFVEdgeNeighborChildFace[i*4+j][2], 
						ncc0, grid);
	if(!f0) {
		std::cout<<"\n [ERROR] no neighbor child face "<<ncc0;
		return;
	}
						
	BccNode3 * f1 = cell1->faceNode(sdb::gdt::TwentyFourFVEdgeNeighborChildFace[i*4+j][2], 
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
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{ 
	BccCell3 * nei = grid->findNeighborCell(cellCoord, i);
	if(!nei) 
		return false;

	return nei->hasChild();
}

bool BccCell3::isEdgeDivided(const int & i,
					const sdb::Coord4 & cellCoord,
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

void BccCell3::findRedValueFrontBlue(BccNode3 * redN,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	//int i = 0;
	//for(;i<4;++i) {
		// BccNode3 * b1 = blueNode(sdb::gdt::FourOppositeBluePair[i][0],
								//cellCoord, grid);
		// BccNode3 * b2 = blueNode(sdb::gdt::FourOppositeBluePair[i][1],
								//	cellCoord, grid);
		//if(b1 && b2) {
		//const IDistanceEdge * eg = fld->edge(b1->index, b2->index);
		//if(eg) 
			//redN->val = eg->val;
		//}
			
	//}
}

bool BccCell3::isFront(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	const float vred = find(15)->val;
	for(int i=0;i<8;++i) {
		if((blueNode(i, cellCoord, grid)->val * vred) < 0.f)
			return true;
	}
	return false;
}

bool BccCell3::isInterior(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid)
{
	const float vred = find(15)->val;
	if(vred > 0.f)
		return false;
		
	for(int i=0;i<8;++i) {
		if((blueNode(i, cellCoord, grid)->val * vred) < 0.f)
			return false;
	}
	return true;
}

}
}
