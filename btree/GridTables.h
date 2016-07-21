/*
 *  GridTables.h
 *  foo
 *
 *  Created by jian zhang on 7/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <Vector3F.h>

namespace aphid {

namespace sdb {

namespace gdt {

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
inline void GetFaceNodeOffset(aphid::Vector3F & dst, const int & i)
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

}

}

}