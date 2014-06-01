/*
 *  TireMesh.cpp
 *  wheeled
 *
 *  Created by jian zhang on 6/1/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "TireMesh.h"

namespace caterpillar {
#define NUMGRIDRAD 56
#define DELTARAD .1121997376282069f
#define NUMGRIDX 9
#define NUMVXH 4
#define NUMVX 10
static const float rportion[NUMVX] = {0.f, 0.382683432365f, 0.707106781187f, 0.923879532511f, 1.f, 
	1.f, 0.923879532511f, 0.707106781187f, 0.382683432365f, 0.f};
	
static const float xportion[NUMVX] = {0.f, 0.076120467489f, 0.292893218813f, 0.617316567635f, 1.f, 
	0.f, 0.382683432365f, 0.707106781187f, 0.923879532511f, 1.f};
	
TireMesh::TireMesh() {}
TireMesh::~TireMesh() {}
btCollisionShape* TireMesh::create(const float & radiusMajor, const float & radiusMinor, const float & width)
{
	const int nv = NUMVX * NUMGRIDRAD;
	btVector3 * pos = createVertexPos(nv);
	const float minX = width * -.5f;
	const float minR = radiusMajor - radiusMinor;
	const float midW = width - radiusMinor * 2.f;
	
	float x, y, z, r, alpha = 0.f;
	int i, j;
	for(j = 0; j < NUMGRIDRAD; j++) {
		for(i = 0; i < NUMVX; i++) {
			r = minR + radiusMinor * rportion[i];
			z = cos(alpha) * r;
			y = sin(alpha) * r;
			x = minX + radiusMinor * xportion[i];
			if(i > NUMVXH) x += midW;
		
			*pos = btVector3(x, y, z);
			pos++;
		}
		alpha += DELTARAD;
	}
	
	const int nt = NUMGRIDX * NUMGRIDRAD * 2;
	int * tri = createTriangles(nt);
	for(j = 0; j < NUMGRIDRAD - 1; j++) {
		for(i = 0; i < NUMGRIDX; i++) {
			*tri = j * NUMVX + i;
			tri++;
			*tri = j * NUMVX + i + 1;
			tri++;
			*tri = (j + 1)* NUMVX + i + 1;
			tri++;
			
			*tri = j * NUMVX + i;
			tri++;
			*tri = (j + 1) * NUMVX + i + 1;
			tri++;
			*tri = (j + 1)* NUMVX + i;
			tri++;
		}
	}
	
	j = NUMGRIDRAD - 1;
	for(i = 0; i < NUMGRIDX; i++) {
		*tri = j * NUMVX + i;
		tri++;
		*tri = j * NUMVX + i + 1;
		tri++;
		*tri = 0 * NUMVX + i + 1;
		tri++;
		
		*tri = j * NUMVX + i;
		tri++;
		*tri = 0 * NUMVX + i + 1;
		tri++;
		*tri = 0 * NUMVX + i;
		tri++;
	}
	
	return createCollisionShape();
}
}