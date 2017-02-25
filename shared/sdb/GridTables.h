/*
 *  GridTables.h
 *  foo
 *
 *  Created by jian zhang on 7/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <math/Vector3F.h>

namespace aphid {

namespace sdb {

namespace gdt {

enum NodePropertyType {
		NUnKnown = 0,
		NFace = 2,
		NBlue = 3,
		NRed = 4,
		NYellow = 5,
		NCyan = 6,
		NRedBlue = 46,
		NRedCyan = 47,
		NRedYellow = 48
};

static const float EightCellChildOffset[8][3] = {
{-1.f, -1.f, -1.f},
{ 1.f, -1.f, -1.f},
{-1.f,  1.f, -1.f},
{ 1.f,  1.f, -1.f},
{-1.f, -1.f,  1.f},
{ 1.f, -1.f,  1.f},
{-1.f,  1.f,  1.f},
{ 1.f,  1.f,  1.f}
};

static const int TwentySixNeighborOffset[26][3] = {
{-1, 0, 0}, // face 0 - 5
{ 1, 0, 0},
{ 0,-1, 0},
{ 0, 1, 0},
{ 0, 0,-1},
{ 0, 0, 1},
{-1,-1,-1}, // vertex 6 - 13
{ 1,-1,-1},
{-1, 1,-1},
{ 1, 1,-1},
{-1,-1, 1},
{ 1,-1, 1},
{-1, 1, 1},
{ 1, 1, 1},
{-1, 0,-1}, // edge 14 - 25
{ 1, 0,-1},
{-1, 0, 1},
{ 1, 0, 1},
{ 0,-1,-1},
{ 0, 1,-1},
{ 0,-1, 1},
{ 0, 1, 1},
{-1,-1, 0},
{ 1,-1, 0},
{-1, 1, 0},
{ 1, 1, 0}
};

/// 3 face, 1 vertex, 3 edge
static const int SevenNeighborOnCorner[8][7] = {
{0, 2, 4,  6, 14, 18, 22},	
{1, 2, 4,  7, 15, 18, 23},
{0, 3, 4,  8, 14, 19, 24},
{1, 3, 4,  9, 15, 19, 25},
{0, 2, 5, 10, 16, 20, 22},
{1, 2, 5, 11, 17, 20, 23},
{0, 3, 5, 12, 16, 21, 24},
{1, 3, 5, 13, 17, 21, 25}
};

/// neighbor coord offset 
/// opposite face in neighbor
static const int SixNeighborOnFace[6][4] = {
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
static const int TwelveBlueBlueEdges[12][5] = {
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
{ 9,13, 913, 1, 3}
};

static const int ThreeNeighborOnEdge[36][4] = {
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

/// for each child 1 red 1 blue 3 edge 3 face 
/// type of node 0 blue 1 edge 2 face 3 red
static const int EightChildBlueInParentTyp[8][8] = {
{0, 1, 1, 2, 1, 2, 2, 3},
{1, 0, 2, 1, 2, 1, 3, 2},
{1, 2, 0, 1, 2, 3, 1, 2},
{2, 1, 1, 0, 3, 2, 2, 1},
{1, 2, 2, 3, 0, 1, 1, 2},
{2, 1, 3, 2, 1, 0, 2, 1},
{2, 3, 1, 2, 1, 2, 0, 1},
{3, 2, 2, 1, 2, 1, 1, 0}
};
/// id of node
static const int EightChildBlueInParentInd[8][8] = {
{ 0, 0, 4, 4, 8, 2, 0,15},
{ 0, 1, 4, 5, 2, 9,15, 1},
{ 4, 4, 2, 1, 0,15,10, 3},
{ 4, 5, 1, 3,15, 1, 3,11},
{ 8, 2, 0,15, 4, 2, 6, 5},
{ 2, 9,15, 1, 2, 5, 5, 7},
{ 0,15,10, 3, 6, 5, 6, 3},
{15, 1, 3,11, 5, 7, 3, 7}
};

static const float TwelveCellLevelLegend[12][3] = {
{1.f, 0.f, 0.f},
{0.f, 1.f, 0.f},
{0.f, 0.f, 1.f},
{0.f, 1.f, 1.f},
{1.f, 0.f, 1.f},
{1.f, 1.f, 0.f},
{1.f, .5f, 0.f},
{.5f, 1.f, 0.f},
{.5f, 0.f, 1.f},
{0.f, .5f, .5f},
{.5f, 0.f, .5f},
{.5f, .5f, 0.f}
};

/// six face, four edge per face
/// vertex ind 0 1
/// edge ind 0 1
/// opposite vertex ind 0 1 in neighbor
static const int TwentyFourFVBlueBlueEdge[24][6] = {
{ 8, 6, 4, 10, 9, 7 }, {12, 8, 10, 6, 13, 11 }, {10,12, 6, 8, 11, 13 }, { 6,10,  8, 4,  7, 11 }, /// -x
{ 7, 9, 5, 9,  6, 8 }, { 9,13, 11, 5,  8, 12 }, {13,11, 7, 11,12, 10 }, {11, 7,  9, 7, 10,  6 }, /// +x
{ 6, 7, 0, 8,  8, 9 }, {10, 6,  8, 2, 12,  8 }, {11,10, 2, 9, 13, 12 }, { 7,11,  9, 0,  9, 13 }, /// -y
{ 9, 8, 1, 11, 7, 6 }, { 8,12, 10, 1,  6, 10 }, {12,13, 3, 10,10, 11 }, {13, 9, 11, 3, 11,  7 }, /// +y
{ 7, 6, 0, 5, 11,10 }, { 9, 7,  5, 1, 13, 11 }, { 8, 9, 1,  4,12, 13 }, { 6, 8,  4, 0, 10, 12 }, /// -z
{10,11, 2, 6,  6, 7 }, {11,13,  7, 2,  7,  9 }, {13,12, 3,  7, 9,  8 }, {12,10,  6, 3,  8,  6 }  /// +z
};

/// per face per edge
/// child i0 i1 child facei
static const int TwentyFourFVEdgeNeighborChildFace[24][3] = {
{3, 1, 1}, /// -x
{7, 3, 1},
{5, 7, 1},
{1, 5, 1},
{0, 2, 0}, /// +x
{2, 6, 0},
{6, 4, 0},
{4, 0, 0},
{2, 3, 3}, /// -y
{6, 2, 3},
{7, 6, 3},
{3, 7, 3},
{1, 0, 2}, /// +y
{0, 4, 2},
{4, 5, 2},
{5, 1, 2},
{5, 4, 5}, /// -z
{7, 5, 5},
{6, 7, 5},
{4, 6, 5},
{0, 1, 4}, /// +z
{1, 3, 4},
{3, 2, 4},
{2, 0, 4},
};

static const int FourOppositeBluePair[4][2] = {
{0, 7},
{1, 6},
{2, 5},
{3, 4}
};

static const float FracOneOverTwoPowers[11] = {
1.f,
.5f,
.25f,
.125f,
.0625f,
.03125f,
.015625f,
.0078125f,
.00390625f,
.001953125f,
.0009765625f
};

#define GDT_FAC_ONEOVER8  .125f
#define GDT_FAC_ONEOVER16  .0625f
#define GDT_FAC_ONEOVER32  .03125f
#define GDT_FAC_ONEOVER64  .015625f
#define GDT_FAC_ONEOVER128  .0078125f

inline bool isEighChildBlueInParentIsBlue(const int & i, const int & j)
{ return EightChildBlueInParentTyp[i][j] == 0; }

inline bool isEighChildBlueInParentIsCyan(const int & i, const int & j)
{ return EightChildBlueInParentTyp[i][j] == 1; }

inline bool isEighChildBlueInParentIsYellow(const int & i, const int & j)
{ return EightChildBlueInParentTyp[i][j] == 2; }

inline bool isEighChildBlueInParentIsRed(const int & i, const int & j)
{ return EightChildBlueInParentTyp[i][j] == 3; }

inline int KeyToBlue(const Vector3F & corner,
				const Vector3F & center)
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

/// i 0:7 j 0:6
inline int GetVertexNeighborJ(const int & i, const int & j)
{ return SevenNeighborOnCorner[i][j]; }

/// i 0:5
inline void GetFaceNodeOffset(Vector3F & dst, const int & i)
{
	dst.x = gdt::TwentySixNeighborOffset[i][0];
	dst.y = gdt::TwentySixNeighborOffset[i][1];
	dst.z = gdt::TwentySixNeighborOffset[i][2];
}

/// i 0:7
inline void GetVertexNodeOffset(aphid::Vector3F & dst, const int & i)
{
	dst.x = gdt::TwentySixNeighborOffset[i+6][0];
	dst.y = gdt::TwentySixNeighborOffset[i+6][1];
	dst.z = gdt::TwentySixNeighborOffset[i+6][2];
}

/// i 0:25
inline void GetNeighborOffset(aphid::Vector3F & dst, const int & i)
{
	dst.x = gdt::TwentySixNeighborOffset[i][0];
	dst.y = gdt::TwentySixNeighborOffset[i][1];
	dst.z = gdt::TwentySixNeighborOffset[i][2];
}

inline void GetCellColor(Vector3F & c, const int & level)
{
	c.set(gdt::TwelveCellLevelLegend[level][0],
			gdt::TwelveCellLevelLegend[level][1],
			gdt::TwelveCellLevelLegend[level][2]); 
}

namespace bcc {
enum NodeColor {
	NUnKnown = 0,
	NFace = 2,
	NBlue = 3,
	NRed = 4,
	NYellow = 5,
	NCyan = 6,
	NRedBlue = 46,
	NRedCyan = 47,
	NRedYellow = 48
};
}

inline void GetNodeColor(float & r, float & g, float & b,
					const int & prop)
{
	switch (prop) {
		case bcc::NFace:
			r = .45f; g = .25f; b = .25f;
			break;
		case bcc::NBlue:
			r = 0.f; g = 0.f; b = 1.f;
			break;
		case bcc::NRed:
			r = 1.f; g = 0.f; b = 0.f;
			break;
		case bcc::NYellow:
			r = .99f; g = 0.99f; b = 0.f;
			break;
		case bcc::NCyan:
			r = 0.f; g = 0.59f; b = 0.89f;
			break;
		case bcc::NRedBlue:
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

}

}

}