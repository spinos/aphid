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

int BccCell::SixTetraFace[6][8] = {
{ 8, 6,12, 8,10,12, 6,10},
{ 7, 9, 9,13,13,11,11, 7},
{ 6, 7,10, 6,11,10, 7,11},
{ 9, 8, 8,12,12,13,13, 9},
{ 7, 6, 9, 7, 8, 9, 6, 8},
{10,11,11,13,13,12,12,10}
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
		if(!grid->findCell(neighborCoord(cellCoord, i) ) ) {
			BccNode * ni = new BccNode;
			ni->key = i;
			
			grid->insert(cellCoord, ni);
		}
	}
	
/// corner 7 - 14 when on border or neighbor has no opposing node
	i=6;
	for(;i<14;++i) {
		
		bool toadd = true;
		neighborOffset(&offset, i);
		samp = m_center + offset * gsize * .5f;
		
		for(j=0;j<7;++j) {
			int neighborJ = SevenNeighborOnCorner[i-6][j];
			neighborOffset(&offset, neighborJ);
			Vector3F neighborCenter = m_center + offset * gsize;
			if(findNeighborCorner(grid, neighborCenter,
					keyToCorner(samp, neighborCenter) ) ) {
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

void BccCell::getNodePositions(Vector3F * dest,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	const float gsize = grid->gridSize();
	sdb::Array<int, BccNode> * cell = grid->findCell(cellCoord);
	if(!cell) {
		std::cout<<"\n [ERROR] no cell "<<cellCoord;
		return;
	}
	
	cell->begin();
	while(!cell->end() ) {
		
		getNodePosition(&dest[cell->value()->index],
					cell->value()->key,
					gsize);
		
		cell->next();
	}
}

void BccCell::connectNodes(std::vector<ITetrahedron *> & dest,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord,
					STriangleArray * faces) const
{
	const float gsize = grid->gridSize();
	sdb::Array<int, BccNode> * cell = grid->findCell(cellCoord);
	if(!cell) {
		std::cout<<"\n [ERROR] no cell "<<cellCoord;
		return;
	}
	
	BccNode * node15 = cell->find(15);
	if(!node15) {
		std::cout<<"\n [ERROR] no node15 ";
		return;
	}
	
/// for each face
	if(!grid->findCell(neighborCoord(cellCoord, 0) ) )
		connectNodesOnFace(dest, grid, cell, cellCoord, node15, 0, faces);
	if(!grid->findCell(neighborCoord(cellCoord, 2) ) )
		connectNodesOnFace(dest, grid, cell, cellCoord, node15, 2, faces);
	if(!grid->findCell(neighborCoord(cellCoord, 4) ) )
		connectNodesOnFace(dest, grid, cell, cellCoord, node15, 4, faces);
		
	connectNodesOnFace(dest, grid, cell, cellCoord, node15, 1, faces);
	connectNodesOnFace(dest, grid, cell, cellCoord, node15, 3, faces);
	connectNodesOnFace(dest, grid, cell, cellCoord, node15, 5, faces);
}

void BccCell::connectNodesOnFace(std::vector<ITetrahedron *> & dest,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					sdb::Array<int, BccNode> * cell,
					const sdb::Coord3 & cellCoord,
					BccNode * node15,
					const int & iface,
					STriangleArray * faces) const
{
	const int inode15 = node15->index;
	BccNode * nodeA = cell->find(iface);
	if(!nodeA) {
		sdb::Array<int, BccNode> * neicell = grid->findCell(neighborCoord(cellCoord, iface) );
		if(!neicell) {
			std::cout<<"\n [ERROR] no shared node"<<iface<<" in neighbor cell ";
			return;
		}
		nodeA = neicell->find(15);
	}
	const int a = nodeA->index;
/// four tetra
	int i=0;
	for(;i<8;i+=2) {
		BccNode * nodeB = cell->find(SixTetraFace[iface][i]);
		if(!nodeB) {
			nodeB = findCornerNodeInNeighbor(SixTetraFace[iface][i],
								grid,
								cellCoord);
		}
		if(!nodeB)
			return;
		
		BccNode * nodeC = cell->find(SixTetraFace[iface][i+1]);
		if(!nodeC) {
			nodeC = findCornerNodeInNeighbor(SixTetraFace[iface][i+1],
								grid,
								cellCoord);
		}
		if(!nodeC)
			return;
		
		int b = nodeB->index;
		int c = nodeC->index;
		
		ITetrahedron * t = new ITetrahedron;
		setTetrahedronVertices(*t, inode15, a, b, c);
		t->index = dest.size();
		dest.push_back(t);
		
/// add four faces
		addFace(faces, a, b, c, t);
		addFace(faces, inode15, a, b, t);
		addFace(faces, inode15, b, c, t);
		addFace(faces, inode15, c, a, t);
	}
}

void BccCell::addFace(STriangleArray * faces,
				int a, int b, int c,
				ITetrahedron * t) const
{
	sdb::Coord3 itri = sdb::Coord3(a, b, c).ordered();
	STriangle<ITetrahedron> * tri = faces->find(itri );
	if(!tri) {
		tri = new STriangle<ITetrahedron>();
		tri->key = itri;
		tri->ta = t;
		
		faces->insert(itri, tri);
	}
	else {
		tri->tb = t;
	}
}

BccNode * BccCell::findCornerNodeInNeighbor(const int & i,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	const float gsize = grid->gridSize();
	
	Vector3F offset;
	neighborOffset(&offset, i);
	Vector3F cornerP = m_center + offset * gsize * .5f;
	
	int j;
	for(j=0;j<7;++j) {
		int neighborJ = SevenNeighborOnCorner[i-6][j];
		neighborOffset(&offset, neighborJ);
		Vector3F neighborCenter = m_center + offset * gsize;
		
		BccNode * node = findNeighborCorner(grid, neighborCenter,
				keyToCorner(cornerP, neighborCenter) );
		if(node) {
			return node;
		}
	}
	std::cout<<"\n [ERROR] no node"<<i<<" in cell"<<cellCoord;
	return NULL;
}

sdb::Coord3 BccCell::neighborCoord(const sdb::Coord3 & cellCoord, int i) const
{
	sdb::Coord3 r = cellCoord;
	r.x += (int)TwentySixNeighborOffset[i][0];
	r.y += (int)TwentySixNeighborOffset[i][1];
	r.z += (int)TwentySixNeighborOffset[i][2];
	return r;
}

void BccCell::neighborOffset(aphid::Vector3F * dest, int i) const
{
	dest->set(TwentySixNeighborOffset[i][0],
					TwentySixNeighborOffset[i][1],
					TwentySixNeighborOffset[i][2]);
}

BccNode * BccCell::findNeighborCorner(sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const Vector3F & pos, int icorner) const
{
	sdb::Array<int, BccNode> * neicell = grid->findCell(pos);
	if(!neicell) 
		return false;
	
	return neicell->find(icorner );
}

int BccCell::keyToCorner(const Vector3F & corner,
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

void BccCell::getNodePosition(aphid::Vector3F * dest,
						const int & nodeI,
						const float & gridSize) const
{
	Vector3F offset;
	neighborOffset(&offset, nodeI);
	offset *= gridSize * .5f;
	if(nodeI == 15)
		offset.set(0.f, 0.f, 0.f);
		
	*dest = m_center + offset;
}

BccTetraGrid::BccTetraGrid() 
{}

BccTetraGrid::~BccTetraGrid() 
{}

void BccTetraGrid::buildNodes()
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

void BccTetraGrid::getNodePositions(Vector3F * dest)
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
		c.getNodePositions(dest, this, gridCoord((const float *)c.centerP() ) );
	}
	cells.clear();
}

void BccTetraGrid::buildTetrahedrons(std::vector<ITetrahedron *> & dest)
{
	std::vector<BccCell> cells;
	begin();
	while(!end() ) {
		cells.push_back(BccCell(coordToCellCenter(key() ) ) );
		next();
	}
	
	STriangleArray faces;
	
	const int n = cells.size();
	int i=0;
	for(;i<n;++i) {
		const BccCell & c = cells[i];
		c.connectNodes(dest, this, gridCoord((const float *)c.centerP() ), &faces );
	}
	cells.clear();
	std::cout<<"\n n face "<<faces.size();
	std::cout.flush();
	
	faces.begin();
	while(!faces.end() ) {
		STriangle<ITetrahedron> * f = faces.value();
		if(f->tb) {
			bool stat = connectTetrahedrons(f->ta, f->tb,
								f->key.x, f->key.y, f->key.z);
			if(!stat) {
				printTetrahedronCannotConnect(f->ta, f->tb);
			}
		}
		faces.next();
	}
	faces.clear();
}

}
