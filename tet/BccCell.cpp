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
/// opposite vertex ind 0 1 in neighbor
int BccCell::TwentyFourFVBlueBlueEdge[24][6] = {
{ 8, 6, 4, 10, 9, 7 }, {12, 8, 10, 6, 13, 11 }, {10,12, 6, 8, 11, 13 }, { 6,10,  8, 4,  7, 11 }, /// -x
{ 7, 9, 5, 9,  6, 8 }, { 9,13, 11, 5,  8, 12 }, {13,11, 7, 11,12, 10 }, {11, 7,  9, 7, 10,  6 }, /// +x
{ 6, 7, 0, 8,  8, 9 }, {10, 6,  8, 2, 12,  8 }, {11,10, 2, 9, 13, 12 }, { 7,11,  9, 0,  9, 13 }, /// -y
{ 9, 8, 1, 11, 7, 6 }, { 8,12, 10, 1,  6, 10 }, {12,13, 3, 10,10, 11 }, {13, 9, 11, 3, 11,  7 }, /// +y
{ 7, 6, 0, 5, 11,10 }, { 9, 7,  5, 1, 13, 11 }, { 8, 9, 1,  4,12, 13 }, { 6, 8,  4, 0, 10, 12 }, /// -z
{10,11, 2, 6,  6, 7 }, {11,13,  7, 2,  7,  9 }, {13,12, 3,  7, 9,  8 }, {12,10,  6, 3,  8,  6 }  /// +z
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

/// eight tetra per face
/// yellow blue cyan
int BccCell::FortyEightTetraFace[48][3] = {
{0, 6, 610}, /// -x 6 10 12 8
{0, 610, 10},
{0, 10, 1012},
{0, 1012, 12},
{0, 12, 812},
{0, 812, 8},
{0, 8, 68},
{0, 68, 6},
{1, 7, 79}, /// +x 7 9 13 11
{1, 79, 9},
{1, 9, 913},
{1, 913, 13},
{1, 13, 1113},
{1, 1113, 11},
{1, 11, 711},
{1, 711, 7},
{2, 6, 67}, /// -y 6 7 11 10
{2, 67, 7},
{2, 7, 711},
{2, 711, 11},
{2, 11, 1011},
{2, 1011, 10},
{2, 10, 610},
{2, 610, 6},
{3, 8, 812}, /// +y 8 12 13 9
{3, 812, 12},
{3, 12, 1213},
{3, 1213, 13},
{3, 13, 913},
{3, 913, 9},
{3, 9, 89},
{3, 89, 8},
{4, 6, 68}, /// -z 6 8 9 7
{4, 68, 8},
{4, 8, 89},
{4, 89, 9},
{4, 9, 79},
{4, 79, 7},
{4, 7, 67},
{4, 67, 6},
{5, 10, 1011}, /// +z 10 11 13 12
{5, 1011, 11},
{5, 11, 1113},
{5, 1113, 13},
{5, 13, 1213},
{5, 1213, 12},
{5, 12, 1012},
{5, 1012, 10}
};

BccCell::BccCell(const Vector3F &center )
{ m_center = center; }

const Vector3F * BccCell::centerP() const
{ return &m_center; }

/// per vertex i 0:7
void BccCell::getCellCorner(aphid::Vector3F & p, const int & i,
					const float & gridSize) const
{
	neighborOffset(&p, i+6);
	p *= gridSize * .5f;
	p += m_center;
}

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
/// negative sides
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
		if(!nodeB) {
			std::cout<<"\n [ERROR] no blue"<<TwentyFourFVBlueBlueEdge[edgei][0];
			return;
		}
		
		BccNode * nodeC = blueNode6(TwentyFourFVBlueBlueEdge[edgei][1],
									cell, grid, cellCoord);
		if(!nodeC) {
			std::cout<<"\n [ERROR] no blue"<<TwentyFourFVBlueBlueEdge[edgei][1];
			return;
		}
		
		int b = nodeB->index;
		int c = nodeC->index;

		BccNode * cyanN = blueBlueNode(TwentyFourFVBlueBlueEdge[edgei][2],
									cell, grid, cellCoord);									
		if(cyanN) {
		
			refineTetrahedron(edgei, 0, iface, dest, faces, inode15, a, b, cyanN->index,
								cell);
			refineTetrahedron(edgei, 1, iface, dest, faces, inode15, a, cyanN->index, c,
								cell);
		}
		else
			addTetrahedron(dest, faces, inode15, a, b, c);
	}
}

/// per face vary edge i 0:23
/// j 0 blue first 1 cyan first
void BccCell::refineTetrahedron(const int & i, const int & j,
					const int & iface,
					std::vector<ITetrahedron *> & dest,
					STriangleArray * faces,
					int ired, int iyellow, int ibc1, int ibc2,
					aphid::sdb::Array<int, BccNode> * cell) const
{
	BlueYellowCyanRefine refiner(ired, iyellow, ibc1, ibc2);
	
	if(j==0) {

		BccNode * redBlueN = cell->find(30000 + TwentyFourFVBlueBlueEdge[i][0]);
		if(redBlueN) {
			//if(iface == 4) std::cout<<" face4 red blue "<<redBlueN->key<<" i "<<i;
			refiner.splitBlue(redBlueN->index);
		}
			
		int iedge = TwentyFourFVBlueBlueEdge[i][2];
		BccNode * redCyanN = cell->find(30000 + TwelveBlueBlueEdges[iedge][2]);
		if(redCyanN) {
			//if(iface == 4) std::cout<<" face4 red cyan "<<redCyanN->key<<" i "<<i;
			refiner.splitCyan(redCyanN->index);
		}
	}
	else {

		BccNode * redBlueN = cell->find(30000 + TwentyFourFVBlueBlueEdge[i][1]);
		if(redBlueN) {
			//if(iface == 4) std::cout<<" face4 red blue "<<redBlueN->key<<" i "<<i;
			refiner.splitCyan(redBlueN->index);
		}
		
		int iedge = TwentyFourFVBlueBlueEdge[i][2];
		BccNode * redCyanN = cell->find(30000 + TwelveBlueBlueEdges[iedge][2]);
		if(redCyanN) {
			//if(iface == 4) std::cout<<" face4 red cyan "<<redCyanN->key<<" i "<<i;
			refiner.splitBlue(redCyanN->index);
		}
	}
	
	BccNode * redYellowN = cell->find(30000 + iface);
	if(redYellowN) {
		//if(iface == 4) std::cout<<" face4 red yellow "<<redYellowN->key;
		refiner.splitYellow(redYellowN->index);
	}
	
	const int nt = refiner.numTetra();
	
	for(int k=0; k<nt; ++k) {
		const ITetrahedron * t = refiner.tetra(k);
		addTetrahedron(dest, faces, t->iv0, t->iv1, t->iv2, t->iv3);
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

/// i 6:13
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

/// per face i 0:5
BccNode * BccCell::yellowNode(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
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

/// blue 6:13
/// cyan 67:1213
BccNode * BccCell::blueOrCyanNode(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	if(i<14)
		return blueNode6(i, cell, grid, cellCoord);
	
	int j = 0;
	switch (i) {
		case 67:
			j=0;
			break;
		case 89:
			j=1;
			break;
		case 1011:
			j=2;
			break;
		case 1213:
			j=3;
			break;
		case 68:
			j=4;
			break;
		case 79:
			j=5;
			break;
		case 1012:
			j=6;
			break;
		case 1113:
			j=7;
			break;
		case 610:
			j=8;
			break;
		case 711:
			j=9;
			break;
		case 812:
			j=10;
			break;
		case 913:
			j=11;
			break;
		default:
			break;
	}
	
	return blueBlueNode(j, cell, grid, cellCoord);
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
	dst.setZero();
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
	ni->prop = -1;
	grid->insert(cellCoord, ni);
	return ni;
}

BccNode * BccCell::addYellowNode(const int & i,
					sdb::WorldGrid<sdb::Array<int, BccNode>, BccNode > * grid,
					const sdb::Coord3 & cellCoord) const
{
	BccNode * ni = new BccNode;
	ni->key = 15000 + i;
	ni->prop = -1;
	grid->insert(cellCoord, ni);
	return ni;
}

bool BccCell::moveBlueTo(Vector3F & p,
					const Vector3F & q,
					const float & r)
{
#if 0
	Vector3F dp = p - q;
	if(Absolute<float>(dp.x) > r) return false;
	if(Absolute<float>(dp.y) > r) return false;
	if(Absolute<float>(dp.z) > r) return false;
	return true;
#else
	Vector3F dp = p - q;
	float d = dp.length();
	if(d > r) {
		dp.normalize();
		p = q + dp * r;
	}
	return d < r;
#endif
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
					const sdb::Coord3 & cellCoord) const
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
					const sdb::Coord3 & cellCoord) const
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

/// connected to two face
/// cyan inside either tetra formed by red-yello-blue-blue
/// keep distance to tetra face
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
	BccNode * yellowN1 = redRedNode(TwelveBlueBlueEdges[i][3],
					cell, grid, cellCoord);
					
	if(Convexity::CheckInsideTetra(redP, yellowN1->pos, p1, p2, p0, r) ) {
		//std::cout<<"\n pass inside tetra check";
		return true;
	}
		
	BccNode * yellowN2 = redRedNode(TwelveBlueBlueEdges[i][4],
					cell, grid, cellCoord);
	
	if(Convexity::CheckInsideTetra(redP, yellowN2->pos, p1, p2, p0, r) ) {
		//std::cout<<"\n pass inside tetra check";
		return true;
	}
	
	return false;
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

/// per face i 0:5
/// find blue-cyan-blue trio on front
int BccCell::faceHasEdgeOnFront(const int & i, 
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					aphid::Vector3F & edgeCenter) const
{
	Vector3F curEdgeC;
	const int edgei = i * 4;
	int c = 0;
/// for each edge
	int j=0;
	for(;j<4;++j) {
		bool edgeOnFront = true;
		curEdgeC.setZero();
		
		BccNode * blueN1 = blueNode6(TwentyFourFVBlueBlueEdge[edgei+j][0],
									cell, grid, cellCoord);
		
		if(blueN1->prop > 0)
			curEdgeC += blueN1->pos;
		else
			edgeOnFront = false;
		
		if(edgeOnFront) {
			
			BccNode * blueN2 = blueNode6(TwentyFourFVBlueBlueEdge[edgei+j][1],
									cell, grid, cellCoord);
			if(blueN2->prop > 0)
				curEdgeC += blueN2->pos;
			else
				edgeOnFront = false;
		}
		
		if(edgeOnFront) {
			c++;
			edgeCenter = curEdgeC * .5f;
		}
	}
	
	return c;
}

/// two yellow on front i 0:2
bool BccCell::oppositeFacesOnFront(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					aphid::Vector3F & pcenter) const
{
	pcenter.setZero();
	Vector3F edgeCenter;
	int cyellow = 0, cblue = 0;
	int j = 0;
	for(;j<2;++j) {

		BccNode * yellowN = redRedNode(ThreeYellowEdge[i][j], cell, grid, cellCoord);
		if(yellowN) {
			
			if(yellowN->prop > 0) {
				pcenter += yellowN->pos;
				cyellow++;
			}
			else {
				if(faceHasEdgeOnFront(ThreeYellowEdge[i][j], cell, grid, cellCoord, edgeCenter) > 0 )
				cblue++;
				pcenter += edgeCenter;
			}
		}
	}
	
	if(cyellow < 1)
		return false;
			
	pcenter *= .5f;
	return (cyellow + cblue) > 1;
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
	
	if(nyellowOnFront > 1) {
		pcenter *= 1.f / (float)nyellow;
		return;
	}

/// must in pairs	
	pcenter.setZero();
	nyellowOnFront = nyellow = 0;

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

bool BccCell::checkMoveRed(const aphid::Vector3F & q,
					const float & r,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	Vector3F p[4];
	int i=0, j;
	for(;i<6;++i) {
		for(j=0;j<4;++j) {
			const int edgei = i * 4 + j;
			BccNode * blueN = blueNode6(TwentyFourFVBlueBlueEdge[edgei][0],
									cell, grid, cellCoord);
			p[j] = blueN->pos;
		}
		if(!Convexity::CheckDistancePlane(p[0], p[1], p[2], p[3], q, r) )
			return false;
	}
	return true;
}

/// per tetra i 0:47 blue cyan end and cut
void BccCell::getFVTetraBlueCyan(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					BccNode ** bc) const
{
	bc[0] = blueOrCyanNode(FortyEightTetraFace[i][1], cell, grid, cellCoord);
	bc[1] = blueOrCyanNode(FortyEightTetraFace[i][2], cell, grid, cellCoord);
	bc[2] = tetraRedBlueCyanYellow(i, 1, cell);
	bc[3] = tetraRedBlueCyanYellow(i, 2, cell);
}


BccNode * BccCell::tetraRedBlueCyanYellow(const int & i,
					const int & j,
					aphid::sdb::Array<int, BccNode> * cell) const
{ return cell->find(30000 + FortyEightTetraFace[i][j]); }

/// i 0:5 j 0:3
BccNode * BccCell::cutFaceVaryBlueCyanYellow(const int & i,
					const int & j,
					BccNode * endN,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					const aphid::Vector3F & q,
					const float & r) const
{
	int k;
	NodePropertyType ntype;
	if(isNodeYellow(endN) ) {
		k = i;
		ntype = NRedYellow;
	}
	else if(isNodeBlue(endN) ) {
		k = TwentyFourFVBlueBlueEdge[i * 4 + j][0];
		ntype = NRedBlue;
	}
	else {
		k = TwentyFourFVBlueBlueEdge[i * 4 + j][2];
		k = TwelveBlueBlueEdges[k][2];
		ntype = NRedCyan;
	}
	
	BccNode * cutN = addNode(30000 + k, cell, grid, cellCoord);
	cutN->pos = q;
	cutN->prop = ntype;
	
	return cutN;
}

/// per tetra i 0:47  
/// j 0:2
BccNode * BccCell::cutTetraRedBlueCyanYellow(const int & i,
					const int & j,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					const Vector3F & q,
					const float & r) const
{
	BccNode * redN = cell->find(15);
	if(redN->pos.distanceTo(q) < r) {
		std::cout<<"\n split close to red";
		redN->pos = q;
		redN->prop = BccCell::NRed;
		return NULL;
	}
	
	BccNode * ycb;
	if(j<1) 
		ycb = yellowNode(FortyEightTetraFace[i][j], cell, grid, cellCoord);
	else 
		ycb = blueOrCyanNode(FortyEightTetraFace[i][j], cell, grid, cellCoord);
		
	if(ycb->pos.distanceTo(q) < r) {
		if(isNodeYellow(ycb) ) {
			std::cout<<"\n split close to yellow";
			ycb->prop = BccCell::NYellow;
			//ycb->pos = q;
		}
		else if(isNodeBlue(ycb) ) {
			std::cout<<"\n split close to blue";
			ycb->prop = BccCell::NBlue;
		}
		else {
			std::cout<<"\n split close to cyan";
			ycb->prop = BccCell::NCyan;
		}
		return NULL;
	}
			
	BccNode * cutN = addNode(30000 + FortyEightTetraFace[i][j], cell, grid, cellCoord);
	cutN->pos = q;
	
	if(isNodeYellow(ycb) )
		cutN->prop = NRedYellow;
	else if(isNodeBlue(ycb) )
		cutN->prop = NRedBlue;
	else
		cutN->prop = NRedCyan;
	
	return cutN;
}

/// move cyan to front
void BccCell::moveTetraCyan(const int & i,
					const int & j,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					const aphid::Vector3F & q) const
{
	BccNode * ycb  = blueOrCyanNode(FortyEightTetraFace[i][j], cell, grid, cellCoord);
	std::cout<<"\n move cyan";
	//ycb->pos = q;
	ycb->prop = NCyan;
}

bool BccCell::isNodeYellow(const BccNode * n) const
{ return (n->key > 14999 && n->key < 15006);}

bool BccCell::isNodeBlue(const BccNode * n) const
{ return (n->key > 5 && n->key < 14); }

BccNode * BccCell::addNode(const int & k,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	BccNode * n = cell->find(k);
	if(!n) {
		n = new BccNode;
		n->key = k;
		n->prop = -1;
		grid->insert(cellCoord, n);
	}
	return n;
}

BccNode * BccCell::orangeNode(const float & i, 
					aphid::sdb::Array<int, BccNode> * cell) const
{ return cell->find(30000 + i); }

BccNode * BccCell::faceVaryRedBlueCutNode(const int & i,
					const int & j,
					aphid::sdb::Array<int, BccNode> * cell) const
{ return cell->find(30000 + TwentyFourFVBlueBlueEdge[i*4+j][0]); }

BccNode * BccCell::faceVaryRedCyanCutNode(const int & i,
					const int & j,
					aphid::sdb::Array<int, BccNode> * cell) const
{ 
	int iedge = TwentyFourFVBlueBlueEdge[i*4+j][2];
	return cell->find(30000 + TwelveBlueBlueEdges[iedge][2]); 
}

BccNode * BccCell::redCyanNode(const int & i, 
					aphid::sdb::Array<int, BccNode> * cell) const
{ return cell->find(30000 + TwelveBlueBlueEdges[i][2]); }

bool BccCell::edgeCrossFront(const BccNode * endN,
						const BccNode * cutN) const
{
	if(cutN) return true;
	return endN->prop > 0;
}

bool BccCell::vertexHasThreeEdgeOnFront(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					aphid::Vector3F & q) const
{
	q.setZero();
	int c = 0;
	int j = 0;
	for(;j<3;++j) {
		BccNode * cyanN = blueBlueNode(EightVVBlueBlueEdge[i][j], cell, grid, cellCoord);
		if(cyanN->prop > 0) {
			c++;
			q += cyanN->pos;
		}
		else {
			int iedge = EightVVBlueBlueEdge[i][j];
			BccNode * cyanCutN = cell->find(30000 + TwelveBlueBlueEdges[i][2]);
			if(cyanCutN) {
				c++;
				q += cyanCutN->pos;
			}
		}
	}
	
	if(c>2)
		q *= .33f;
	return c>2;
}

bool BccCell::vertexHasThreeFaceOnFront(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	int c = 0;
	int j = 3;
	for(;j<6;++j) {
		BccNode * yellowN = yellowNode(EightVVBlueBlueEdge[i][j], cell, grid, cellCoord);
		if(yellowN->prop > 0)
			c++;
	}
	return c>2;
}

bool BccCell::edgeHasTwoFaceOnFront(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					aphid::Vector3F & q) const
{
	BccNode * yellowN1 = yellowNode(TwenlveEdgeYellowInd[i][0], cell, grid, cellCoord);
	if(yellowN1->prop < 0)
		return false;
		
	q = yellowN1->pos;
		
	BccNode * yellowN2 = yellowNode(TwenlveEdgeYellowInd[i][1], cell, grid, cellCoord);
	if(yellowN2->prop < 0)
		return false;
			
	q += yellowN2->pos;
	q *= .5f;
	return true;
}

BccNode * BccCell::cutBlue(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					const aphid::Vector3F & q) const
{
	BccNode * cutN = addNode(30000 + i + 6, cell, grid, cellCoord);
	cutN->pos = q;
	cutN->prop = NRedBlue;
	return cutN;
}

BccNode * BccCell::cutCyan(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					const aphid::Vector3F & q) const
{
	BccNode * cutN = addNode(30000 + TwelveBlueBlueEdges[i][2], cell, grid, cellCoord);
	cutN->pos = q;
	cutN->prop = NRedCyan;
	return cutN;
}

void BccCell::getBlueMean(aphid::Vector3F & q,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	float s = 0.f;
	q.setZero();
	int i = 6;
	for(;i<14;++i) {
		BccNode * bn = blueNode6(i, cell, grid, cellCoord);
		if(bn->prop < 0) {
			q += bn->pos * .5f;
			s += .5f;
		}
		else {
			q += bn->pos * .25f;
			s += .25f;
		}
	}
	q /= s;
}

void BccCell::GetNodeColor(float & r, float & g, float & b,
					const int & prop)
{
	switch (prop) {
		case NBlue:
			r = 0.f; g = 0.f; b = 1.f;
			break;
		case NRed:
			r = 1.f; g = 0.f; b = 0.f;
			break;
		case NYellow:
			r = .99f; g = 0.99f; b = 0.f;
			break;
		case NCyan:
			r = 0.f; g = 0.59f; b = 0.89f;
			break;
		case NRedBlue:
			r = 0.79f; g = 0.f; b = 0.89f;
			break;
		case -4:
			r = 0.3f; g = 0.f; b = 0.f;
			break;
		default:
			r = g = b = .3f;
			break;
	}
}

bool BccCell::snapToFront(BccNode * a, BccNode * b) const
{
	const float la = Absolute<float>(a->val);
	const float lb = Absolute<float>(b->val);
	const float l = la + lb;
	const float h = l * .107f;
	
	Vector3F v = b->pos - a->pos;
	
	if(la < h) {
		a->val = 0.f;
		a->pos += v * (la / l);  
		return true;
	}
	
	if(lb < h) {
		b->val = 0.f;
		b->pos -= v * (lb / l); 
		return true;
	}
	
	return false;
}

void BccCell::getSplitPos(aphid::Vector3F & dst,
					BccNode * a, BccNode * b) const
{
	float sa = Absolute<float>(a->val);
	float sb = Absolute<float>(b->val);
	dst = a->pos * (sb / (sa + sb)) + b->pos * (sa / (sa + sb));
}

void BccCell::cutSignChangeEdge(aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	int i, k;
/// per edge
	i = 0;
	for(;i<12;++i) {
		BccNode * cyanN = blueBlueNode(i, cell, grid, cellCoord);
		if(cyanN)
			continue;
			
		BccNode * b1N = blueBlueEdgeNode(i, 0, cell, grid, cellCoord);
		BccNode * b2N = blueBlueEdgeNode(i, 1, cell, grid, cellCoord);
		if(b1N->val * b2N->val < 0.f) {
			if(snapToFront(b1N, b2N) ) {
				if(b1N->val == 0.f)
					b1N->prop = NBlue;
				else 
					b2N->prop = NBlue;
			}
			else {
				cyanN = addEdgeNode(i, grid, cellCoord);
				cyanN->prop = NCyan;
				getSplitPos(cyanN->pos, b1N, b2N); 
				cyanN->val = 0.f;
				cyanN->index = -1;
			}
		}
	}
		
	BccNode * redN = cell->find(15);
	
	if(redN->prop == NRed)
		return;
		
/// per face
	i = 0;
	for(;i<6;++i) {
		BccNode * yellowN = yellowNode(i, cell, grid, cellCoord);
		if(yellowN)
			continue;
		
		BccNode * faN = faceNode(i, cell, grid, cellCoord);
		
		if(redN->val * faN->val < 0.f) {
			if(snapToFront(redN, faN) ) {
				if(redN->val == 0.f) {
					redN->prop = NRed;
					return;
				}
				else 
					faN->prop = NRed;
			}
			else {
				yellowN = addYellowNode(i, grid, cellCoord);
				yellowN->prop = NYellow;
				getSplitPos(yellowN->pos, redN, faN); 
				yellowN->val = 0.f;
				yellowN->index = -1;
			}
		}
	}
		
/// per vertex
	i = 0;
	for(;i<8;++i) {
		k = keyToRedBlueCut(i+6);
		BccNode * redBlueN = cell->find(k);
		if(redBlueN)
			continue;
			
		BccNode * blueN = blueNode(i+6, cell, grid, cellCoord);
		if(redN->val * blueN->val < 0.f) {
			if(snapToFront(redN, blueN) ) {
				if(blueN->val == 0.f)
					blueN->prop = NBlue;
			}
			else {
				redBlueN = addNode(k, cell, grid, cellCoord);
				redBlueN->prop = NRedBlue;
				getSplitPos(redBlueN->pos, redN, blueN); 
				redBlueN->val = 0.f;
				redBlueN->index = -1;
			}
		}
	}

}

void BccCell::cutFVTetraAuxEdge(const int & i,
					const int & j,
					const BccNode * nodeA,
					const BccNode * nodeB,
					RedBlueRefine & refiner,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	const int edgei = i * 4 + j;
	BccNode * nodeC = blueNode6(TwentyFourFVBlueBlueEdge[edgei][0],
									cell, grid, cellCoord);
	BccNode * nodeD = blueNode6(TwentyFourFVBlueBlueEdge[edgei][1],
									cell, grid, cellCoord);
	refiner.set(nodeA->index, nodeB->index, nodeC->index, nodeD->index);
	refiner.evaluateDistance(nodeA->val, nodeB->val, nodeC->val, nodeD->val);
	if(!refiner.hasOption() )
		return;
		
/// yellow
	if(refiner.needSplitRedEdge(0) ) {
		BccNode * yellowN = yellowNode(i, cell, grid, cellCoord);
		if(!yellowN) {
			yellowN = addYellowNode(i, grid, cellCoord);
			yellowN->index = -1;
			yellowN->pos = refiner.splitPos(nodeA->val, nodeB->val, nodeA->pos, nodeB->pos);
			yellowN->prop = -1;
		}
	}
	
/// cyan
	if(refiner.needSplitRedEdge(1) ) {
		BccNode * cyanN = faceVaryBlueBlueNode(i, j, cell, grid, cellCoord);
		if(!cyanN) {
			cyanN = addFaceVaryEdgeNode(i, j, grid, cellCoord);
			cyanN->index = -1;
			cyanN->pos = refiner.splitPos(nodeC->val, nodeD->val, nodeC->pos, nodeD->pos);
			cyanN->prop = -1;
		}
	}

/// red blue	
	if(refiner.needSplitBlueEdge(0) ) {
		int k = TwentyFourFVBlueBlueEdge[edgei][0];
		k = keyToRedBlueCut(k);
		BccNode * redBlueN = cell->find(k);
		if(!redBlueN) {
			redBlueN = addNode(k, cell, grid, cellCoord);
			redBlueN->index = -1;
			redBlueN->pos = refiner.splitPos(nodeA->val, nodeC->val, nodeA->pos, nodeC->pos);
			redBlueN->prop = -1;
		}
	}
	
	if(refiner.needSplitBlueEdge(1) ) {
		int k = TwentyFourFVBlueBlueEdge[edgei][1];
		k = keyToRedBlueCut(k);
		BccNode * redBlueN = cell->find(k);
		if(!redBlueN) {
			redBlueN = addNode(k, cell, grid, cellCoord);
			redBlueN->index = -1;
			redBlueN->pos = refiner.splitPos(nodeA->val, nodeD->val, nodeA->pos, nodeD->pos);
			redBlueN->prop = -1;
		}
	}
	
/// far side red blue in neighbor cell
	const sdb::Coord3 neiC = neighborCoord(cellCoord, i);
	sdb::Array<int, BccNode> * neiCell = grid->findCell(neiC);
	
	if(refiner.needSplitBlueEdge(2) ) {
		if(neiCell) {
			int k = TwentyFourFVBlueBlueEdge[edgei][4];
			k = keyToRedBlueCut(k);
			BccNode * redBlueN = neiCell->find(k);
			if(!redBlueN) {
				redBlueN = addNode(k, neiCell, grid, neiC);
				redBlueN->index = -1;
				redBlueN->pos = refiner.splitPos(nodeB->val, nodeC->val, nodeB->pos, nodeC->pos);
				redBlueN->prop = -1;
			}
		}
		else {
			int k = TwentyFourFVBlueBlueEdge[edgei][0];
			k = keyToFaceBlueCut(i, k);
			BccNode * redBlueN = cell->find(k);
			if(!redBlueN) {
				redBlueN = addNode(k, cell, grid, cellCoord);
				redBlueN->index = -1;
				redBlueN->pos = refiner.splitPos(nodeB->val, nodeC->val, nodeB->pos, nodeC->pos);
				redBlueN->prop = -1;
			}
		}
	}
	
	if(refiner.needSplitBlueEdge(3) ) {
		if(neiCell) {
			int k = TwentyFourFVBlueBlueEdge[edgei][5];
			k = keyToRedBlueCut(k);
			BccNode * redBlueN = neiCell->find(k);
			if(!redBlueN) {
				redBlueN = addNode(k, neiCell, grid, neiC);
				redBlueN->index = -1;
				redBlueN->pos = refiner.splitPos(nodeB->val, nodeD->val, nodeB->pos, nodeD->pos);
				redBlueN->prop = -1;
			}
		}
		else {
			int k = TwentyFourFVBlueBlueEdge[edgei][1];
			k = keyToFaceBlueCut(i, k);
			BccNode * redBlueN = cell->find(k);
			if(!redBlueN) {
				redBlueN = addNode(k, cell, grid, cellCoord);
				redBlueN->index = -1;
				redBlueN->pos = refiner.splitPos(nodeB->val, nodeD->val, nodeB->pos, nodeD->pos);
				redBlueN->prop = -1;
			}
		}
	}
}

void BccCell::cutAuxEdge(RedBlueRefine & refiner,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	const BccNode * redN = cell->find(15);
	
/// per face
	int i = 0, j;
	for(;i<6;++i) {
		BccNode * faN = faceNode(i, cell, grid, cellCoord);
/// negative side checked by previous cell
		//if((i&1) == 0 && faN->key == 15)
		//	continue;
						
/// per tetra
		for(j=0;j<4;++j) {
			cutFVTetraAuxEdge(i, j, redN, faN, refiner, cell, grid, cellCoord);
		}
	}
}

void BccCell::connectTetrahedrons(std::vector<ITetrahedron *> & dest,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	BccNode * redN = cell->find(15);
	int i = 0, j;
/// per face
	for(;i<6;++i) {
		BccNode * faN = faceNode(i, cell, grid, cellCoord);
		if((i&1) == 0) {
			if(faN->key == 15)
				continue;
		}
/// per edge
		for(j=0;j<4;++j) {
			BccNode * cN = blueNode6(TwentyFourFVBlueBlueEdge[i*4+j][0],
									cell, grid, cellCoord);
			BccNode * dN = blueNode6(TwentyFourFVBlueBlueEdge[i*4+j][1],
									cell, grid, cellCoord);
									
			ITetrahedron * t = new ITetrahedron;
			resetTetrahedronNeighbors(*t);
			setTetrahedronVertices(*t, redN->index, faN->index, cN->index, dN->index);
			t->index = dest.size();
			dest.push_back(t);
		}
	}
}

void BccCell::cutFVRefinerEdges(const int & i, const int & j,
					const BccNode * nodeA, const BccNode * nodeB, 
					const BccNode * nodeC, const BccNode * nodeD, 
					RedBlueRefine & refiner,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	if(!refiner.hasOption() ) 
		return;
		
/// yellow
	if(refiner.needSplitRedEdge(0) ) {
		BccNode * yellowN = yellowNode(i, cell, grid, cellCoord);
		if(yellowN)
			refiner.splitRedEdge(0, yellowN->index, yellowN->pos);
		else
			std::cout<<"\n [ERROR] no yellow "<<cellCoord;
	}
	
/// cyan
	if(refiner.needSplitRedEdge(1) ) {
		BccNode * cyanN = faceVaryBlueBlueNode(i, j, cell, grid, cellCoord);
		if(cyanN) 
			refiner.splitRedEdge(1, cyanN->index, cyanN->pos);
		else
			std::cout<<"\n [ERROR] no cyan "<<cellCoord;
	}
	
	const int edgei = i * 4 + j;

/// red blue	
	if(refiner.needSplitBlueEdge(0) ) {
		int k = TwentyFourFVBlueBlueEdge[edgei][0];
		k = keyToRedBlueCut(k);
		BccNode * redBlueN = cell->find(k);
		if(redBlueN) 
			refiner.splitBlueEdge(0, redBlueN->index, redBlueN->pos);
		else
			std::cout<<"\n [ERROR] no red blue0 "<<cellCoord;
	}
	
	if(refiner.needSplitBlueEdge(1) ) {
		int k = TwentyFourFVBlueBlueEdge[edgei][1];
		k = keyToRedBlueCut(k);
		BccNode * redBlueN = cell->find(k);
		if(redBlueN) {
			refiner.splitBlueEdge(1, redBlueN->index, redBlueN->pos);
		}
		else
			std::cout<<"\n [ERROR] no red blue1 "<<cellCoord;
	}
	
/// far side red blue in neighbor cell
	const sdb::Coord3 neiC = neighborCoord(cellCoord, i);
	sdb::Array<int, BccNode> * neiCell = grid->findCell(neiC);
	
	if(refiner.needSplitBlueEdge(2) ) {
		if(neiCell) {
			int k = TwentyFourFVBlueBlueEdge[edgei][4];
			k = keyToRedBlueCut(k);
			BccNode * redBlueN = neiCell->find(k);
			if(redBlueN) 
				refiner.splitBlueEdge(2, redBlueN->index, redBlueN->pos);
			else
				std::cout<<"\n [ERROR] no red blue2 "<<neiC;
		}
		else {
			int k = TwentyFourFVBlueBlueEdge[edgei][0];
			k = keyToFaceBlueCut(i, k);
			BccNode * redBlueN = cell->find(k);
			if(redBlueN) 
				refiner.splitBlueEdge(2, redBlueN->index, redBlueN->pos);
			else
				std::cout<<"\n [ERROR] no red blue2 "<<cellCoord;
		}
	}
	
	if(refiner.needSplitBlueEdge(3) ) {
		if(neiCell) {
			int k = TwentyFourFVBlueBlueEdge[edgei][5];
			k = keyToRedBlueCut(k);
			BccNode * redBlueN = neiCell->find(k);
			if(redBlueN) 
				refiner.splitBlueEdge(3, redBlueN->index, redBlueN->pos);
			else
				std::cout<<"\n [ERROR] no red blue3 "<<neiC;
		}
		else {
			int k = TwentyFourFVBlueBlueEdge[edgei][1];
			k = keyToFaceBlueCut(i, k);
			BccNode * redBlueN = cell->find(k);
			if(redBlueN) {
				refiner.splitBlueEdge(3, redBlueN->index, redBlueN->pos);
			}
			else
				std::cout<<"\n [ERROR] no red blue3 "<<cellCoord;
		}
	}
	
	refiner.refine();
	if(!refiner.checkTetraVolume() )
		refiner.verbose();
}

void BccCell::connectRefinedTetrahedrons(std::vector<ITetrahedron *> & dest,
					RedBlueRefine & refiner,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const
{
	BccNode * redN = cell->find(15);
	int i = 0, j, k, nt;
/// per face
	for(;i<6;++i) {
		BccNode * faN = faceNode(i, cell, grid, cellCoord);
		if((i&1) == 0) {
			if(faN->key == 15)
				continue;
		}
/// per edge
		for(j=0;j<4;++j) {
			BccNode * cN = blueNode6(TwentyFourFVBlueBlueEdge[i*4+j][0],
									cell, grid, cellCoord);
			BccNode * dN = blueNode6(TwentyFourFVBlueBlueEdge[i*4+j][1],
									cell, grid, cellCoord);
			
			refiner.set(redN->index, faN->index, cN->index, dN->index);
			refiner.evaluateDistance(redN->val, faN->val, cN->val, dN->val);
			refiner.estimateNormal(redN->pos, faN->pos, cN->pos, dN->pos);
			cutFVRefinerEdges(i, j, 
									redN, faN, cN, dN,
									refiner, cell, grid, cellCoord);
			
			nt = refiner.numTetra();
			for(k=0;k<nt;++k) {
				ITetrahedron * t = new ITetrahedron;
				resetTetrahedronNeighbors(*t);
				
				const ITetrahedron * bt = refiner.tetra(k);
				
				setTetrahedronVertices(*t, bt->iv0, bt->iv1, bt->iv2, bt->iv3);
				t->index = dest.size();
				dest.push_back(t);
			}
		}
	}
}

int BccCell::keyToFaceBlueCut(const int & i, const int & j) const
{ return 40000 + i * 10000 + j; }

int BccCell::keyToRedBlueCut(const int & i) const
{ return 30000 + i; }

}