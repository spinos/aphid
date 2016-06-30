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
{-1.f, 0.f, 0.f}, // face 0 - 5
{ 1.f, 0.f, 0.f},
{ 0.f,-1.f, 0.f},
{ 0.f, 1.f, 0.f},
{ 0.f, 0.f,-1.f},
{ 0.f, 0.f, 1.f},
{-1.f,-1.f,-1.f}, // vertex 6 - 13
{ 1.f,-1.f,-1.f},
{-1.f, 1.f,-1.f},
{ 1.f, 1.f,-1.f},
{-1.f,-1.f, 1.f},
{ 1.f,-1.f, 1.f},
{-1.f, 1.f, 1.f},
{ 1.f, 1.f, 1.f},
{-1.f, 0.f,-1.f}, // edge 14 - 25
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
		resetTetrahedronNeighbors(*t);
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

bool BccCell::moveNode(int & xi,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord,
					const Vector3F & p,
					int * moved) const
{
	int k = keyToCorner(p, m_center );
	Vector3F offset;
	neighborOffset(&offset, k);
	Vector3F vp = m_center + offset * grid->gridSize() * .5f;
	
/// choose center of vertex
	if(m_center.distanceTo(p) < vp.distanceTo(p) )
		k = 15;
	
	if(moved[k])
		return false;
	
	sdb::Array<int, BccNode> * cell = grid->findCell(cellCoord);
		
	if(k==15) {
		
		BccNode * node15 = cell->find(15);
		xi = node15->index;
	}
	else {
		return false;
		BccNode * nodeK = cell->find(k);
		if(!nodeK) {
			nodeK = findCornerNodeInNeighbor(k,
								grid,
								cellCoord);
		}
		xi = nodeK->index;
	}
	moved[k] = 1;
	return true;
}

int BccCell::indexToNode15(aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const	
{
	sdb::Array<int, BccNode> * cell = grid->findCell(cellCoord );
		
	BccNode * node15 = cell->find(15);
	return node15->index;
}

/// vertex node
/// i 0:7
int BccCell::indexToBlueNode(const int & i,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					const float & cellSize,
					aphid::Vector3F & p) const
{
	sdb::Array<int, BccNode> * cell = grid->findCell(cellCoord );
	BccNode * node = cell->find(i+6);
	if(!node) 
		node = findCornerNodeInNeighbor(i+6,
								grid,
								cellCoord);
	
	Vector3F offset;
	neighborOffset(&offset, i+6);
	p = m_center + offset * cellSize * .5f;
	return node->index;
}

bool BccCell::moveNode15(int & xi,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord,
					const Vector3F & p) const
{
	int k = keyToCorner(p, m_center );
	Vector3F offset;
	neighborOffset(&offset, k);
	Vector3F vp = m_center + offset * grid->gridSize() * .5f;
	
/// choose center of vertex
	if(m_center.distanceTo(p) > vp.distanceTo(p) )
		return false;
	
	sdb::Array<int, BccNode> * cell = grid->findCell(cellCoord);
		
	BccNode * node15 = cell->find(15);
	xi = node15->index;
	return true;
}

bool BccCell::getVertexNodeIndices(int vi, int * xi,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord)
{
	sdb::Array<int, BccNode> * cell = grid->findCell(cellCoord);
	BccNode * nodeK = cell->find(vi);
	if(!nodeK) {
		return false;
	}
	xi[0] = nodeK->index;
	
	BccNode * node15 = cell->find(15);
	xi[1] = node15->index;
	
	const float gsize = grid->gridSize();
	
	Vector3F offset;
	neighborOffset(&offset, vi);
	
	int j;
	for(j=0;j<7;++j) {
		int neighborJ = SevenNeighborOnCorner[vi-6][j];
		neighborOffset(&offset, neighborJ);
		Vector3F neighborCenter = m_center + offset * gsize;
		
		sdb::Array<int, BccNode> * nei = grid->findCell((const float *)&neighborCenter);
		if(!nei) {
			return false;
		}
		
		BccNode * nei15 = nei->find(15);
		xi[2+j] = nei15->index;
	}
	return true;
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

void BccTetraGrid::moveNodeIn(const aphid::Vector3F & cellCenter,
					const Vector3F * pos, 
					const int & n,
					aphid::Vector3F * X,
					int * prop)
{
/// each node can move once
	int moved[16];
	int i = 0;
	for(;i<16;++i)
		moved[i] = 0;
		
	BccCell cell(cellCenter );
	
	Vector3F closestP;
	float minD = 1e8f;
	float d;
	for(i=0;i<n;++i) {
		d = cellCenter.distanceTo(pos[i]);
		if(d<minD) {
			minD = d;
			closestP = pos[i];
		}
	}
	
	int xi;
#if 1
	if(cell.moveNode(xi, this, gridCoord((const float *)&cellCenter ),
					closestP,
					moved) ) {
		X[xi] = closestP;
		prop[xi] = 1;
	}
#else
	for(i=0;i<n;++i) {
		if(cell.moveNode(xi, this, gridCoord((const float *)&cellCenter ),
					pos[i],
					moved) ) {
			X[xi] = pos[i];
			prop[xi] = 1;
		}
	}
#endif
}

bool BccTetraGrid::moveRedNodeTo(const Vector3F & cellCenter,
					const float & d,
					const Vector3F & pos,
					Vector3F * X,
					int * prop)
{	
	BccCell cell(cellCenter);
	int xi = cell.indexToNode15(this, gridCoord((const float *)&cellCenter ) );
			
/// limit in which red can move
	if(d > gridSize() * .25f) {
		return false;
	}
		
	X[xi] = pos;
	prop[xi] = 4;
	
	return true;
}

void BccTetraGrid::moveBlueNodes(const Vector3F & cellCenter,
					const aphid::Vector3F & redP,
					const std::vector<Vector3F> & samples,
					aphid::Vector3F * X,
					int * prop)
{
	const float gz = gridSize();
	const float r = gz * .25f;
	BccCell cell(cellCenter);
	const sdb::Coord3 cellCoord = gridCoord((const float *)&cellCenter );
	int xi;
	float d;
	Vector3F blueP, closestP;
/// for each blue
	int i=0;
	for(;i<8;++i) {
		xi = cell.indexToBlueNode(i, this, cellCoord, gz, blueP);
/// already moved
		if(prop[xi] > 0) 
			continue;
			
		GetClosest<Vector3F>(closestP, d, blueP, samples);
		
		if(isBlueCloseToRed(closestP, blueP, redP, r) ) {
			continue;
		}
			
		//std::cout<<"\n move blue"<<xi;
		X[xi] = closestP;
		prop[xi] = 3;
	}
}

/// limit minimum angles
bool BccTetraGrid::isBlueCloseToRed(const aphid::Vector3F & p,
					const aphid::Vector3F & blueP,
					const aphid::Vector3F & redP,
					const float & r) const
{
/// limit in octan
	Vector3F dp = p - blueP;
	if(Absolute<float>(dp.x) > 2.f * r) return true;
	if(Absolute<float>(dp.y) > 2.f * r) return true;
	if(Absolute<float>(dp.z) > 2.f * r) return true;
/// limit to red
	dp = p - redP;
	if(Absolute<float>(dp.x) < r) return true;
	if(Absolute<float>(dp.y) < r) return true;
	if(Absolute<float>(dp.z) < r) return true;
	return false;
}

void BccTetraGrid::moveRedNodeIn(const aphid::Vector3F & cellCenter,
					const Vector3F & pos,
					aphid::Vector3F * X,
					int * prop)
{
	BccCell cell(cellCenter);
	int xi;
	if(cell.moveNode15(xi, this, gridCoord((const float *)&cellCenter ),
					pos) ) {
		X[xi] = pos;
		prop[xi] = 1;
	}
}

void BccTetraGrid::smoothBlueNodeIn(const aphid::Vector3F & cellCenter,
					aphid::Vector3F * X)
{
	BccCell cell(cellCenter);
	int xi[9];
	Vector3F pos;
/// for each vertex node
	int i = 6;
	for(;i<14;++i) {
		if(cell.getVertexNodeIndices(i, xi, this, gridCoord((const float *)&cellCenter ) ) ) {
/// average of eight nodes
			X[xi[0]] = (X[xi[1]]
						+ X[xi[2]]
						+ X[xi[3]]
						+ X[xi[4]]
						+ X[xi[5]]
						+ X[xi[6]]
						+ X[xi[7]]
						+ X[xi[8]]) * .125f;
		}
	}
}

}
