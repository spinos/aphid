/*
 *  BccCell.cpp
 *  foo
 *
 *  Created by jian zhang on 7/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BccCell.h"

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

int BccCell::SixNeighborOnFace[6][4] = {
{-1, 0, 0, 1},
{ 1, 0, 0, 0},
{ 0,-1, 0, 3},
{ 0, 1, 0, 2},
{ 0, 0,-1, 5},
{ 0, 0, 1, 4}
};

int BccCell::TwelveBlueBlueEdges[12][3] = {
{ 6, 7, 67},
{ 8, 9, 89},
{10,11, 1011},
{12,13, 1213},
{ 6, 8, 68},
{ 7, 9, 79},
{10,12, 1012},
{11,13, 1113},
{ 6,10, 610},
{ 7,11, 711},
{ 8,12, 812},
{ 9,13, 913},
};

int BccCell::ThreeNeighborOnEdge[36][4] = {
{ 0, 0,-1, 1011}, { 0,-1,-1, 1213}, { 0,-1, 0, 89  },
{ 0, 0,-1, 1213}, { 0, 1,-1, 1011}, { 0, 1, 0, 67  },
{ 0, 0, 1, 67  }, { 0,-1, 1, 89  }, { 0,-1, 0, 1213},
{ 0, 0, 1, 89  }, { 0, 1, 1, 67  }, { 0, 1, 0, 1011},
{-1, 0, 0, 79  }, {-1, 0,-1, 1113}, { 0, 0,-1, 1012},
{ 1, 0, 0, 68  }, { 1, 0,-1, 1012}, { 0, 0,-1, 1113},
{-1, 0, 0, 1113}, {-1, 0, 1, 79  }, { 0, 0, 1, 68  },
{ 1, 0, 0, 1012}, { 1, 0, 1, 68  }, { 0, 0, 1, 79  },
{-1, 0, 0, 711 }, {-1,-1, 0, 913 }, { 0,-1, 0, 812 },
{ 1, 0, 0, 610 }, { 1,-1, 0, 812 }, { 0,-1, 0, 913 },
{-1, 0, 0, 913 }, {-1, 1, 0, 711 }, { 0, 1, 0, 610 },
{ 1, 0, 0, 812 }, { 1, 1, 0, 610 }, { 0, 1, 0, 711 }
};

BccCell::BccCell(const Vector3F &center )
{ m_center = center; }

const Vector3F * BccCell::centerP() const
{ return &m_center; }

void BccCell::addRedBlueNodes(sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
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
			ni->prop = -1;
			getNodePosition(&ni->pos, i, gsize);
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
			ni->prop = -1;
			getNodePosition(&ni->pos, i, gsize);
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
	
	int ired = node15->index;

/// for each face	
	int i=0;
	for(;i<6;++i) {
		
		BccNode * nodeA = redRedNode(i, cell, grid, cellCoord);
		if(!nodeA) {
			if((i & 1)==0) {
				if(grid->findCell(neighborCoord(cellCoord, i) ) )
					continue;
			}
			nodeA = faceNode(i, cell, grid, cellCoord);
		}
		connectNodesOnFace(dest, grid, cell, cellCoord, 
							ired, nodeA->index, i, faces);

	}
}

void BccCell::connectNodesOnFace(std::vector<ITetrahedron *> & dest,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					sdb::Array<int, BccNode> * cell,
					const sdb::Coord3 & cellCoord,
					int inode15,
					int a,
					const int & iface,
					STriangleArray * faces) const
{
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
	if(nodeI == 15) {
		*dest = m_center;
		return;
	}
	
	Vector3F offset;
	neighborOffset(&offset, nodeI);
	offset *= gridSize * .5f;
		
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

/// center node
BccNode * BccCell::redNode(aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const	
{
	sdb::Array<int, BccNode> * cell = grid->findCell(cellCoord );
		
	return cell->find(15);
}

/// vertex node
/// i 0:7
BccNode * BccCell::blueNode(const int & i,
					sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					aphid::Vector3F & p) const
{
	BccNode * node = cell->find(i+6);
	if(!node) 
		node = findCornerNodeInNeighbor(i+6,
								grid,
								cellCoord);
	p = node->pos;
	return node;
}

/// red-red cut
/// i 0:5
BccNode * BccCell::redRedNode(const int & i,
					sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	BccNode * node = cell->find(15000 + i);
	if(!node)
		node = findRedRedNodeInNeighbor(i, 
								grid,
								cellCoord);
		
	return node;
}

/// i 0:5
BccNode * BccCell::findRedRedNodeInNeighbor(const int & i,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	sdb::Coord3 neiC(cellCoord.x + SixNeighborOnFace[i][0],
					cellCoord.y + SixNeighborOnFace[i][1],
					cellCoord.z + SixNeighborOnFace[i][2]);
	
	sdb::Array<int, BccNode> * cell = grid->findCell(neiC);
	if(!cell) 
		return NULL;
	
	return cell->find(15000 + SixNeighborOnFace[i][3]);
}

/// i 0:5
BccNode * BccCell::faceNode(const int & i,
					sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	BccNode * node = cell->find(i);
	if(!node) {
		sdb::Coord3 neiC(cellCoord.x + SixNeighborOnFace[i][0],
					cellCoord.y + SixNeighborOnFace[i][1],
					cellCoord.z + SixNeighborOnFace[i][2]);
		sdb::Array<int, BccNode> * nei = grid->findCell(neiC);
		if(!nei) {
			std::cout<<"\n [ERROR] no neighbor"<<i<<" in cell"<<cellCoord;
			return NULL;
		}
		node = nei->find(15);
	}

	return node;
}

/// looped by four blue
bool BccCell::faceClosed(const int & i,
					sdb::Array<int, BccNode> * cell,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	int j=0;
	for(;j<4;++j) {
		BccNode * nodeB = cell->find(SixTetraFace[i][j*2]);
		if(!nodeB) {
			nodeB = findCornerNodeInNeighbor(SixTetraFace[i][j*2],
								grid,
								cellCoord);
		}
		if(nodeB->prop < 0) 
			return false;
	}
	return true;
}

/// average of four blue
void BccCell::facePosition(Vector3F & dst,
					const int & i,
					sdb::Array<int, BccNode> * cell,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	dst.set(0.f, 0.f, 0.f);
	int j=0;
	for(;j<4;++j) {
		BccNode * nodeB = cell->find(SixTetraFace[i][j*2]);
		if(!nodeB) {
			nodeB = findCornerNodeInNeighbor(SixTetraFace[i][j*2],
								grid,
								cellCoord);
		}
		dst += nodeB->pos;
	}
	dst *= .25f;
}

/// i 0:5
const Vector3F BccCell::facePosition(const int & i, const float & gz) const
{
	Vector3F offset;
	neighborOffset(&offset, i);
	return m_center + offset * gz * .5f;
}

BccNode * BccCell::addFaceNode(const int & i,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord)
{
	BccNode * ni = new BccNode;
	ni->key = 15000 + i;
	grid->insert(cellCoord, ni);
	return ni;
}

bool BccCell::moveBlueTo(const Vector3F & p,
					const Vector3F & q,
					const float & r)
{
	Vector3F dp = p - q;
	if(Absolute<float>(dp.x) >= r) return false;
	if(Absolute<float>(dp.y) >= r) return false;
	if(Absolute<float>(dp.z) >= r) return false;
	return true;
}

bool BccCell::moveFaceTo(const int & i,
					const Vector3F & p,
					const Vector3F & q,
					const Vector3F & redP,
					const float & r)
{
	Vector3F vp = p - q;
	float d = vp.length();
	if(d < .1f * r || d > r) return false;
	
	return (p - redP).length() > r;
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
	
	p = node->pos;
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

/// i 0:11
void BccCell::blueBlueEdgeV(int & v1,
					int & v2,
					const int & i) const
{
	v1 = TwelveBlueBlueEdges[i][0] - 6;
	v2 = TwelveBlueBlueEdges[i][1] - 6;
}

BccNode * BccCell::addEdgeNode(const int & i,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord)
{
	BccNode * ni = new BccNode;
	ni->key = TwelveBlueBlueEdges[i][2];
	grid->insert(cellCoord, ni);
	return ni;
}

/// blue-blue cut
/// i 0:11
BccNode * BccCell::blueBlueNode(const int & i,
					sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	BccNode * node = cell->find(TwelveBlueBlueEdges[i][2]);
	if(!node)
		node = findBlueBlueNodeInNeighbor(i, 
								grid,
								cellCoord);
		
	return node;
}

BccNode * BccCell::findBlueBlueNodeInNeighbor(const int & i,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	int j = 0;
	for(;j<3;++j) {
		sdb::Coord3 neiC(cellCoord.x + ThreeNeighborOnEdge[i*3+j][0],
					cellCoord.y + ThreeNeighborOnEdge[i*3+j][1],
					cellCoord.z + ThreeNeighborOnEdge[i*3+j][2]);
					
		sdb::Array<int, BccNode> * cell = grid->findCell(neiC);
		if(cell) {
			BccNode * node = cell->find(ThreeNeighborOnEdge[i*3+j][3]);
			if(node)
				return node;
		}
	}
	return NULL;
}

}