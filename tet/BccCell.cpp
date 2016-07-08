/*
 *  BccCell.cpp
 *  foo
 *
 *  Created by jian zhang on 7/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BccCell.h"
#include <line_math.h>
#include <tetrahedron_math.h>
#include "Convexity.h"

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

/// six face, four edge per face
/// vertex ind 0 1
/// edge ind 0 1
int BccCell::TwentyFourFVBlueBlueEdge[24][4] = {
{ 8, 6, 4, 10}, {12, 8, 10, 6 }, {10,12, 6, 8 }, { 6,10, 8, 4 }, /// -x
{ 7, 9, 5, 9 }, { 9,13, 11, 5 }, {13,11, 7, 11}, {11, 7, 9, 7 }, /// +x
{ 6, 7, 0, 8 }, {10, 6, 8, 2  }, {11,10, 2, 9 }, { 7,11, 9, 0 }, /// -y
{ 9, 8, 1, 11}, { 8,12, 10, 1 }, {12,13, 3, 10}, {13, 9, 11, 3}, /// +y
{ 7, 6, 0, 5 }, { 9, 7, 5, 1  }, { 8, 9, 1, 4 }, { 6, 8, 4, 0 }, /// -z
{10,11, 2, 6 }, {11,13, 7, 2  }, {13,12, 3, 7 }, {12,10, 6, 3 }  /// +z
};

/// red-blue ind
/// face
/// ind in neighbor
int BccCell::RedBlueEdge[24][3] = {
{15008, 0, 15009}, {15012, 0, 15013}, {15010, 0, 15011}, {15006, 0, 15007},
{15007, 1, 15006}, {15019, 1, 15008}, {15013, 1, 15012}, {15011, 1, 15010},
{15006, 2, 15008}, {15010, 2, 15012}, {15011, 2, 15013}, {15007, 2, 15009},
{15009, 3, 15007}, {15008, 3, 15006}, {15012, 3, 15010}, {15013, 3, 15011},
{15007, 4, 15011}, {15009, 4, 15013}, {15008, 4, 15012}, {15006, 4, 15010},
{15010, 5, 15006}, {15011, 5, 15007}, {15013, 5, 15009}, {15012, 5, 15008}
};

int BccCell::SixTetraFace[6][8] = {
{ 8, 6,12, 8,10,12, 6,10},
{ 7, 9, 9,13,13,11,11, 7},
{ 6, 7,10, 6,11,10, 7,11},
{ 9, 8, 8,12,12,13,13, 9},
{ 7, 6, 9, 7, 8, 9, 6, 8},
{10,11,11,13,13,12,12,10}
};

/// neighbor coord offset 
/// opposite face in neighbor
int BccCell::SixNeighborOnFace[6][4] = {
{-1, 0, 0, 1},
{ 1, 0, 0, 0},
{ 0,-1, 0, 3},
{ 0, 1, 0, 2},
{ 0, 0,-1, 5},
{ 0, 0, 1, 4}
};

/// vertex 0 1
/// edge ind
/// face ind 0 1
int BccCell::TwelveBlueBlueEdges[12][5] = {
{ 6, 7, 67, 2, 4}, /// x
{ 8, 9, 89, 3, 4},
{10,11, 1011, 2, 5},
{12,13, 1213, 3, 5},
{ 6, 8, 68, 0, 4}, /// y
{ 7, 9, 79, 1, 4},
{10,12, 1012, 0, 5},
{11,13, 1113, 1, 5},
{ 6,10, 610, 0, 2}, /// z
{ 7,11, 711, 1, 2},
{ 8,12, 812, 0, 3},
{ 9,13, 913, 1, 3},
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

/// per edge yellow ind in cell and neighbor
int BccCell::TwenlveEdgeYellowInd[12][7] = {
{2, 4, 3, 5, 0, -1, -1}, /// 67
{3, 4, 2, 5, 0,  1, -1}, /// 89
{2, 5, 3, 4, 0, -1,  1}, /// 1011
{3, 5, 2, 4, 0,  1,  1}, /// 1213
{0, 4, 1, 5,-1,  0, -1}, /// 68
{1, 4, 0, 5, 1,  0, -1}, /// 79
{0, 5, 1, 4,-1,  0,  1}, /// 1012
{1, 5, 0, 4, 1,  0,  1}, /// 1113
{0, 2, 1, 3,-1, -1,  0}, /// 610
{1, 2, 0, 3, 1, -1,  0}, /// 711
{0, 3, 1, 2,-1,  1,  0}, /// 812
{1, 3, 0, 2, 1,  1,  0}, /// 913
};

/// per-vertex blue-blue edge and face ind
int BccCell::EightVVBlueBlueEdge[8][6] = {
{0, 4, 8, 0, 2, 4}, /// 67 68 610
{0, 5, 9, 1, 2, 4}, /// 67 79 711
{1, 4,10, 0, 3, 4}, /// 68 89 812
{1, 5,11, 1, 3, 4}, /// 79 89 913
{2, 6, 8, 0, 2, 5}, /// 610 1011 1012
{2, 7, 9, 1, 2, 5}, /// 1011 711 1113 
{3, 6,10, 0, 3, 5}, /// 812 1012 1213 
{3, 7,11, 1, 3, 5}  /// 913 1113 1213
};

int BccCell::ThreeYellowFace[3][4] = {
{0, 1, 2, 3},
{0, 1, 4, 5},
{2, 3, 4, 5}
};

int BccCell::ThreeYellowEdge[3][2] = {
{0, 1},
{2, 3},
{4, 5}
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

/// c-------b
///  \    / |
///   \  /  |
///    red--a
/// iface 0:5
void BccCell::connectNodesOnFace(std::vector<ITetrahedron *> & dest,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					sdb::Array<int, BccNode> * cell,
					const sdb::Coord3 & cellCoord,
					int inode15,
					int a,
					const int & iface,
					STriangleArray * faces) const
{
/// for each edge
	int i=0;
	for(;i<4;++i) {
		const int edgei = iface * 4 + i;
		BccNode * nodeB = blueNode6(TwentyFourFVBlueBlueEdge[edgei][0],
									cell, grid, cellCoord);
		if(!nodeB)
			return;
		
		BccNode * nodeC = blueNode6(TwentyFourFVBlueBlueEdge[edgei][1],
									cell, grid, cellCoord);
		if(!nodeC)
			return;
		
		int b = nodeB->index;
		int c = nodeC->index;
		
		BccNode * nodeD = blueBlueNode(TwentyFourFVBlueBlueEdge[edgei][2],
									cell, grid, cellCoord);									
		if(nodeD) {
/// split into two
			addTetrahedron(dest, faces, inode15, a, b, nodeD->index);
			addTetrahedron(dest, faces, inode15, a, nodeD->index, c);
		}
		else
			addTetrahedron(dest, faces, inode15, a, b, c);
	}
}

void BccCell::addTetrahedron(std::vector<ITetrahedron *> & dest,
						STriangleArray * faces,
						int v0, int a, int b, int c) const
{
	ITetrahedron * t = new ITetrahedron;
	resetTetrahedronNeighbors(*t);
	setTetrahedronVertices(*t, v0, a, b, c);
	t->index = dest.size();
	dest.push_back(t);
	
/// add four faces
	addFace(faces, a, b, c, t);
	addFace(faces, v0, a, b, t);
	addFace(faces, v0, b, c, t);
	addFace(faces, v0, c, a, t);
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

/// per face i 0:5
BccNode * BccCell::neighborRedNode(int i,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	sdb::Array<int, BccNode> * cell = grid->findCell(neighborCoord(cellCoord, i) );
	if(!cell)
		return NULL;
		
	return cell->find(15);
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

BccNode * BccCell::blueNode(const int & i,
					sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	BccNode * node = cell->find(i);
	if(!node) 
		node = findCornerNodeInNeighbor(i,
								grid,
								cellCoord);
	return node;
}

/// i 6:13
BccNode * BccCell::blueNode6(const int & i,
					sdb::Array<int, BccNode> * cell,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	BccNode * node = cell->find(i);
	if(!node) 
		node = findCornerNodeInNeighbor(i,
								grid,
								cellCoord);
	if(!node) 
		std::cout<<"\n [ERROR] cannot find blue node"<<i
				<<" in cell"<<cellCoord;
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

/// looped by four blue on front or blue straddled two blue-blue cut
/// i 0:5
bool BccCell::faceClosed(const int & i,
					sdb::Array<int, BccNode> * cell,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord,
					Vector3F & center) const
{
	center.setZero();
	Vector3F p0, p1;
	int ns;
	int j=0;
	for(;j<4;++j) {
		BccNode * nodeB = faceVaryBlueNode(i, j, cell, grid, cellCoord);

		if(nodeB->prop < 0) {
			ns = straddleBlueCut(i, j, cell, grid, cellCoord, p0, p1);
			if(ns < 2)
				return false;
			
				center += (p0 + p1) * .5f;
		}
		else {
			center += nodeB->pos;
		}
	}
	center *= .25f;
	return true;
}

/// for each face
/// i 0:5
bool BccCell::anyBlueCut(const int & i,
					sdb::Array<int, BccNode> * cell,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	int j=0;
	for(;j<4;++j) {
		const int edgei = i * 4 + j;		
		BccNode * nodeB = blueBlueNode(TwentyFourFVBlueBlueEdge[edgei][2],
									cell, grid, cellCoord);
		if(nodeB)
			return true;
	}
	return false;
}

/// average of four blue
/// cache blue and blue-blue
/// i 0:5
/// ps[8]
/// np 4
/// n node on front
void BccCell::facePosition(Vector3F & dst,
					Vector3F * ps,
					int & np,
					int & numOnFront,
					const int & i,
					sdb::Array<int, BccNode> * cell,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	bool stat = anyBlueCut(i, cell, grid, cellCoord);
		
	numOnFront = 0;
	dst.set(0.f, 0.f, 0.f);
	np = 0;
	int j=0;
	for(;j<4;++j) {
		const int edgei = i * 4 + j;
		
		if(stat) {
			BccNode * nodeC = blueBlueNode(TwentyFourFVBlueBlueEdge[edgei][2],
									cell, grid, cellCoord);
			if(nodeC) {
				dst += nodeC->pos;
				ps[np++] = nodeC->pos;
				if(nodeC->prop > 0)
					numOnFront++;
			}
			else {
				std::cout<<"\n [ERROR] blue cut not looped "<<cellCoord;
			}
		}
		else {
			BccNode * nodeB = blueNode6(TwentyFourFVBlueBlueEdge[edgei][0],
									cell, grid, cellCoord);
			dst += nodeB->pos;
			ps[np++] = nodeB->pos;
			if(nodeB->prop > 0)
				numOnFront++;
		}
		
	}
	dst *= 1.f / (float)np;
}

void BccCell::faceEdgePostion(Vector3F & p1,
					Vector3F & p2,
					const int & iface, 
					const int & iedge,
					sdb::Array<int, BccNode> * cell,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	BccNode * nodeA = blueNode6(TwentyFourFVBlueBlueEdge[iface *4 + iedge][0],
									cell, grid, cellCoord);
	p1 = nodeA->pos;
	
	BccNode * nodeB = blueNode6(TwentyFourFVBlueBlueEdge[iface *4 + iedge][1],
									cell, grid, cellCoord);
	p2 = nodeB->pos;
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

/// i 0:11
BccNode * BccCell::addEdgeNode(const int & i,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord)
{
	BccNode * ni = new BccNode;
	ni->key = TwelveBlueBlueEdges[i][2];
	ni->prop = -1;
	grid->insert(cellCoord, ni);
	return ni;
}

/// i 0:5
/// j 0:3
BccNode * BccCell::addFaceVaryEdgeNode(const int & i,
					const int & j,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord)
{
	int k = TwentyFourFVBlueBlueEdge[i *4 + j][2];
	return addEdgeNode(k, grid, cellCoord);
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

/// red-blue cut
/// i 0:23
BccNode * BccCell::redBlueNode(const int & i,
					sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	return cell->find(RedBlueEdge[i][0]);
}

/// i 0:5
/// j 0:3
BccNode * BccCell::faceVaryBlueNode(const int & i,
					const int & j,
					sdb::Array<int, BccNode> * cell,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	return blueNode6(TwentyFourFVBlueBlueEdge[i *4 + j][0],
									cell, grid, cellCoord);
}

/// i 0:5
/// j 0:3
BccNode * BccCell::faceVaryRedBlueNode(const int & i,
					const int & j,
					sdb::Array<int, BccNode> * cell,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{ return redBlueNode(i *4 + j, cell, grid, cellCoord); }

/// i 0:5
/// j 0:3
BccNode * BccCell::faceVaryBlueBlueNode(const int & i,
					const int & j,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	BccNode * nodeA = blueBlueNode(TwentyFourFVBlueBlueEdge[i *4 + j][2],
									cell, grid, cellCoord);
	return nodeA;
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

/// i 0:23
BccNode * BccCell::findRedBlueNodeInNeighbor(const int & i,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	const int c = RedBlueEdge[i][1];
	sdb::Coord3 neiC(cellCoord.x + SixNeighborOnFace[c][0],
					cellCoord.y + SixNeighborOnFace[c][1],
					cellCoord.z + SixNeighborOnFace[c][2]);
					
	sdb::Array<int, BccNode> * cell = grid->findCell(neiC);
	if(!cell) 
		return NULL;
		
	return cell->find(RedBlueEdge[i][2]);
}

/// p0 inside box p1 p2 r but not close to p0 p1
bool BccCell::checkSplitEdge(const Vector3F & p0,
					const Vector3F & p1,
					const Vector3F & p2,
					const float & r,
					const int & comp) const
{		
	float dts;
	if(!distancePointLineSegment(dts, p0, p1, p2) )
		return false;
		
	if(dts > r)
		return false;
		
	BoundingBox bx;
	bx.expandBy(p1, r);
	bx.expandBy(p2, r);
	if(!bx.isPointInside(p0) )
		return false;
		
	if(p0.distanceTo(p1) < r) 
		return false;
		
	if(p0.distanceTo(p2) < r) 
		return false;
		
	if(comp==0) {
		if(p0.x < p1.x + r
			|| p0.x > p2.x - r)
				return false;
	}
	else if(comp==1) {
		if(p0.y < p1.y + r
			|| p0.y > p2.y - r)
				return false;
	}
	else {
		if(p0.z < p1.z + r
			|| p0.z > p2.z - r)
				return false;
	}
	
	return true;
}

bool BccCell::checkSplitFace(const Vector3F & p0,
					const Vector3F & p1,
					const Vector3F & p2,
					const float & r,
					const int & d,
					const Vector3F * ps,
					const int & np) const
{
	int i = 0, i1, i0;
	for(;i<4;++i) {
		i0 = i-1;
		if(i0<0)
			i0 = 3;
			
		i1 = i+1;
		if(i1>3)
			i1 = 0;
			
		if(!Convexity::CheckDistanceTwoPlanes(p1, ps[i], ps[i0], ps[i1], p0, .3f * r) )
			return false;
		
		if(!Convexity::CheckDistanceFourPoints(p1, ps[i], ps[i0], ps[i1], p0, r) )
			return false;
			
		if(!Convexity::CheckDistanceTwoPlanes(p2, ps[i], ps[i0], ps[i1], p0, .3f * r) )
			return false;
		
		if(!Convexity::CheckDistanceFourPoints(p2, ps[i], ps[i0], ps[i1], p0, r) )
			return false;

/*			
/// low volume
		float tvol = tetrahedronVolume1(p0, p1, ps[i], ps[i1]);
		if(Absolute<float>(tvol) < 1e-2f) {
			//std::cout<<"\n tvol"<<tvol;
			return false;
		}
		
		tvol = tetrahedronVolume1(p0, p2, ps[i], ps[i1]);
		if(Absolute<float>(tvol) < 1e-2f) {
			//std::cout<<"\n tvol"<<tvol;
			return false;
		}
*/
		
	}
/*
	Vector3F q1 = p1;
	Vector3F q2 = p2;
	if(p1.comp(d) > p2.comp(d) ) {
		q1 = p2;
		q2 = p1;
	}
	if(!checkSplitEdge(p0, q1, q2, r, d) )
		return false;
*/
	return true;
}

/// for each blue-blue edge, test distance to red-red edge
bool BccCell::checkWedgeFace(const aphid::Vector3F & p1,
					const aphid::Vector3F & p2,
					const aphid::Vector3F * corners,
					const float & r) const
{
	int i=0, j;
	for(;i<4;++i) {
		j = i+1;
		if(j>3)
			j = 0;
			
		if(distanceBetweenSkewLines(p1, p2, corners[i], corners[j] ) < r )
			return true;
	}
	return false;
}

/// three face per vertex
/// i 0:7
int BccCell::blueNodeFaceOnFront(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					bool & onFront)
{
	int c = 0;
	int j = 0;
	for(;j<3;++j) {
/// adjunct blue-blue cut on front
		BccNode * n = blueBlueNode(EightVVBlueBlueEdge[i][j], cell, grid, cellCoord);
		if(n) {
			if(n->prop > 0)
				c++;
		}
	}
	
	for(j=3;j<6;++j) {
		BccNode * rr = redRedNode(EightVVBlueBlueEdge[i][j], cell, grid, cellCoord);
		if(rr) {
			if(rr->prop > 0)
				c++;
		}
	}
	return c;
}

void BccCell::blueNodeConnectToFront(int & nedge, int & nface,
					const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord)
{
	nedge = nface = 0;
	int j = 0;
	for(;j<3;++j) {
/// adjunct blue-blue cut on front
		BccNode * bn = blueBlueNode(EightVVBlueBlueEdge[i][j], cell, grid, cellCoord);
		if(bn) {
			if(bn->prop > 0)
				nedge++;
		}
	}
	for(j=3;j<6;++j) {
		BccNode * rr = redRedNode(EightVVBlueBlueEdge[i][j], cell, grid, cellCoord);
		if(rr) {
			if(rr->prop > 0)
				nface++;
		}
	}
}

/// i 0:5
/// j 0:3
BccNode * BccCell::addFaceVaryRedBlueEdgeNode(const int & i,
					const int & j,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord)
{
	BccNode * ni = new BccNode;
	ni->key = RedBlueEdge[i*4+j][0];
	ni->prop = -1;
	grid->insert(cellCoord, ni);
	return ni;
}

int BccCell::straddleBlueCut(const int & i,
					const int & j,
					sdb::Array<int, BccNode> * cell,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord,
					Vector3F & p0, Vector3F & p1) const
{
	int c = 0;
	int fve = i * 4 + j;
	BccNode * b0 = blueBlueNode(TwentyFourFVBlueBlueEdge[fve][2], cell, grid, cellCoord);
	if(b0) {
		c++;
		p0 = b0->pos;
	}
	
	BccNode * b1 = blueBlueNode(TwentyFourFVBlueBlueEdge[fve][3], cell, grid, cellCoord);
	if(b1) {
		c++;
		p1 = b1->pos;
	}
	return c;
}

/// i 0:11 j 0:1
BccNode * BccCell::blueBlueEdgeNode(int i, int j,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	return blueNode(TwelveBlueBlueEdges[i][j],
					cell, grid, cellCoord);
}

/// cut edge cannot interset two face
/// find two red
/// keep split tetra volume positive
/// i 0:11
bool BccCell::checkSplitBlueBlueEdge(const aphid::Vector3F & p0,
					const aphid::Vector3F & redP,
					const aphid::Vector3F & p1,
					const aphid::Vector3F & p2,
					const float & r,
					const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	BccNode * antiRedN1 = neighborRedNode(TwelveBlueBlueEdges[i][3],
						grid, cellCoord);
	if(!antiRedN1)
		return false;
	
	//BccNode * yellowN1 = faceNode(TwelveBlueBlueEdges[i][3],
	//				cell, grid, cellCoord);
					
	//if(!Convexity::CheckTetraVolume(redP, yellowN1->pos, p0, p1) )
	//	return false;
		
	//if(!Convexity::CheckTetraVolume(antiRedN1->pos, yellowN1->pos, p1, p0) )
	//	return false;
			
	/*if(!Convexity::CheckDistanceTwoPlanes(redP, yellowN1->pos, p1, p2, p0, .29f * r) )
		return false;
		
	if(!Convexity::CheckDistanceFourPoints(redP, yellowN1->pos, p1, p2, p0, 1.1f * r) )
		return false;*/
		
	BccNode * antiRedN2 = neighborRedNode(TwelveBlueBlueEdges[i][4],
						grid, cellCoord);
	if(!antiRedN2)
		return false;
		
	BccNode * yellowN2 = faceNode(TwelveBlueBlueEdges[i][4],
					cell, grid, cellCoord);
	
	if(!Convexity::CheckTetraVolume(redP, yellowN2->pos, p0, p2) )
		return false;
		
	if(!Convexity::CheckTetraVolume(antiRedN2->pos, yellowN2->pos, p2, p0) )
		return false;
		
	/*if(!Convexity::CheckDistanceTwoPlanes(redP, yellowN2->pos, p1, p2, p0, .29f * r) )
		return false;
		
	if(!Convexity::CheckDistanceFourPoints(redP, yellowN2->pos, p1, p2, p0, 1.1f * r) )
		return false;*/
	
	return true;
}

bool BccCell::checkFaceValume(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	BccNode * redN = cell->find(15);
	Vector3F redP = redN->pos;
	BccNode * yellowN = faceNode(i,
					cell, grid, cellCoord);
	Vector3F yellowP = yellowN->pos;
	
	int j = 0, j1;
	for(;j<4;++j) {
		j1 = j+1;
		if(j1>3) j1 = 0;
		
		BccNode * blueN0 = faceVaryBlueNode(i, j, cell, grid, cellCoord);
		BccNode * blueN1 = faceVaryBlueNode(i, j1, cell, grid, cellCoord);
		
		float tvol = tetrahedronVolume1(redP, yellowP, blueN0->pos, blueN1->pos);
		
		if(Absolute<float>(tvol) < 1e-2f) 
			std::cout<<"\n low tetra vol "<<tvol;
	}
	return true;
}

/// four yellow on front i 0:2
bool BccCell::yellowFaceOnFront(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					aphid::Vector3F & pcenter) const
{//return false;
	pcenter.setZero();
	int j = 0;
	for(;j<4;++j) {
		BccNode * yellowN = redRedNode(ThreeYellowFace[i][j], cell, grid, cellCoord);
		if(!yellowN)
			return false;
			
		if(yellowN->prop < 0)
			return false;
			
		pcenter += yellowN->pos;
	}
	pcenter *= .25f;
	return true;
}

/// two yellow on front i 0:2
bool BccCell::yellowEdgeOnFront(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					aphid::Vector3F & pcenter) const
{return false;
	pcenter.setZero();
	int j = 0;
	for(;j<2;++j) {
		BccNode * yellowN = redRedNode(ThreeYellowEdge[i][j], cell, grid, cellCoord);
		if(!yellowN)
			return false;
			
		if(yellowN->prop < 0)
			return false;
			
		pcenter += yellowN->pos;
	}
	pcenter *= .5f;
	return true;
}

/// per edge i 0:11
/// average four red
void BccCell::edgeRedCenter(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					aphid::Vector3F & pcenter,
					int & nred,
					int & redOnFront) const
{
	nred = 1;
	redOnFront = 0;
	BccNode * redN = cell->find(15);
	pcenter = redN->pos;
	if(redN->prop > 0)
		redOnFront++;
		
	int j=0;
	for(;j<3;++j) {
		sdb::Coord3 neiC(cellCoord.x + ThreeNeighborOnEdge[i*3+j][0],
					cellCoord.y + ThreeNeighborOnEdge[i*3+j][1],
					cellCoord.z + ThreeNeighborOnEdge[i*3+j][2]);
					
		sdb::Array<int, BccNode> * cellj = grid->findCell(neiC);
		if(cellj) {
		BccNode * redjN = cellj->find(15);
		pcenter += redjN->pos;
		nred++;
		if(redjN->prop > 0)
			redOnFront++;
		}
	}
	
	pcenter *= 1.f / (float)nred;
}

/// per edge i 0:11
/// find two yellow pair on front
void BccCell::edgeYellowCenter(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					aphid::Vector3F & pcenter,
					int & nyellow,
					int & nyellowOnFront) const
{
	nyellowOnFront = nyellow = 0;
	
	sdb::Coord3 neiC(cellCoord.x + TwenlveEdgeYellowInd[i][4],
					cellCoord.y + TwenlveEdgeYellowInd[i][5],
					cellCoord.z + TwenlveEdgeYellowInd[i][6]);
	sdb::Array<int, BccNode> * cellnei = grid->findCell(neiC);
	if(!cellnei) 
		return;
		
	pcenter.setZero();
		
	BccNode * yellowN1 = redRedNode(TwenlveEdgeYellowInd[i][0], cell, grid, cellCoord);
	if(yellowN1) {
		pcenter += yellowN1->pos;
		nyellow++;
		if(yellowN1->prop > 0)
			nyellowOnFront++;
	}

	BccNode * yellowN2 = redRedNode(TwenlveEdgeYellowInd[i][2], cellnei, grid, neiC);
	if(yellowN2) {
		pcenter += yellowN2->pos;
		nyellow++;
		if(yellowN2->prop > 0)
			nyellowOnFront++;
	}
	
/// must in pairs
	if(nyellowOnFront < 2) {
		pcenter.setZero();
		nyellowOnFront = 0;
	}

	BccNode * yellowN3 = redRedNode(TwenlveEdgeYellowInd[i][1], cell, grid, cellCoord);
	if(yellowN3) {
		pcenter += yellowN3->pos;
		nyellow++;
		if(yellowN3->prop > 0)
			nyellowOnFront++;
	}

	BccNode * yellowN4 = redRedNode(TwenlveEdgeYellowInd[i][3], cellnei, grid, neiC);
	if(yellowN4) {
		pcenter += yellowN4->pos;
		nyellow++;
		if(yellowN4->prop > 0)
			nyellowOnFront++;
	}
	
	pcenter *= 1.f / (float)nyellow;
}

}