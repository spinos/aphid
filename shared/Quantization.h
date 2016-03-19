/*
 *  Quantization.h
 *  testntree
 *
 *  Created by jian zhang on 3/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Vector3F.h"
#include <cmath>
namespace aphid {

namespace colnor30 {

/// normal-color packed into int bit layout
/// 0-4 color red		5bit
/// 5-9 color green		5bit
/// 10-14 color blue	5bit
/// 15 normal sign		1bit
/// 16-17 normal axis	2bit
/// 18-23 normal u		6bit
/// 24-29 normal v		6bit

void encodeN(int & dst, const Vector3F & n)
{
/// 0 - 5
	int o = n.orientation();
/// 0 - 2
	int axis = o >> 1;
/// 0: - 1: +
	int d = o - axis * 2;
	int u, v;
	float m;

	if(axis ==0) {
		m = d > 0 ? n.x : -n.x;
		u = 32 + 31 * n.y / m;
		v = 32 + 31 * n.z / m;
	}
	else if(axis == 1) {
		m = d > 0 ? n.y : -n.y;
		u = 32 + 31 * n.x / m;
		v = 32 + 31 * n.z / m;
	}
	else {
		m = d > 0 ? n.z : -n.z;
		u = 32 + 31 * n.x / m;
		v = 32 + 31 * n.y / m;
	}
	dst = dst | (d<<15 | axis<<16 | u<<18 | v<<24);
}

void encodeC(int & dst, const Vector3F & c)
{
	int r = c.x < 1.f ? 32 * c.x : 31;
	int g = c.y < 1.f ? 32 * c.y : 31;
	int b = c.z < 1.f ? 32 * c.z : 31;
	
	dst = dst | (r | g<<5 | b<<10);
}

void decodeN(Vector3F & n, const int & src)
{
	int d = src & (1<<15);
	int axis = (src>>16) & 3;
	int u = (src>>18) & 63;
	int v = (src>>24) & 63;
	if(axis ==0) {
		n.y = (float)(u - 32) / 31.f;
		n.z = (float)(v - 32) / 31.f;
		n.x = 1.f;
		if(d==0) n.x = -1.f;
	}
	else if(axis == 1) {
		n.x = (float)(u - 32) / 31.f;
		n.z = (float)(v - 32) / 31.f;
		n.y = 1.f;
		if(d==0) n.y = -1.f;
	}
	else {
		n.x = (float)(u - 32) / 31.f;
		n.y = (float)(v - 32) / 31.f;
		n.z = 1.f;
		if(d==0) n.z = -1.f;
	}
	n.normalize();
}

void decodeC(Vector3F & c, const int & src)
{
	int r = src & 31;
	int g = (src>>5) & 31;
	int b = (src>>10) & 31;
	
	c.x = (float)r/32.f;
	c.y = (float)g/32.f;
	c.z = (float)b/32.f;
}

}

}