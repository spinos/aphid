/*
 *  BladeMesh.cpp
 *  
 *
 *  Created by jian zhang on 8/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BladeMesh.h"

namespace aphid {

BladeMesh::BladeMesh()
{}

BladeMesh::~BladeMesh()
{}

void BladeMesh::createBlade(const float& width, const float& height,
					const float& ribWidth, const float& tipHeight,
					const int& m, const int& n)
{
	float halfWidth = width * .5f;
	
	int halfN = n>>1;
	float deltaX;
	if(n > 2)
		deltaX = (width - ribWidth) / (float)(n-2);
	else
		deltaX = width / (float)(n-1);
	
	float ribHeight = height - tipHeight;
		
	float deltaY = ribHeight / (float)m;
	float deltaTipY = (height - ribHeight) /(float)halfN; 

	LoftMeshBuilder bld;
	
	Vector3F pv(0.f, 0.f, 0.f);
/// add points in rows
	for(int j=0;j<m+1;++j) {
		pv.x = -halfWidth;
		for(int i=0;i<n;++i) {
			
			bld.addPoint(pv);
/// has rib
			if(n > 2) {
				if(i== halfN - 1)
					pv.x += ribWidth;
				else
					pv.x += deltaX;
			}
		}
		pv.y += deltaY;
	}
	
/// tip points
	pv.x = 0.f;
	pv.y = ribHeight + deltaTipY;
	
	for(int j=0;j<halfN;++j) {
		
		bld.addPoint(pv);
		
		pv.y += deltaTipY;
	}
	
	int* profvs = new int[m + 2];
	
	for(int i=0;i<n;++i) {
		for(int j=0;j<m+1;++j) {
			profvs[j] = j * n + i;
		}
		
		int side = i - halfN;
		if(side < 0)
			side = -(side + 1);
			
		profvs[m+1] = (m+1) * n + side;
		
		bld.addProfile(m + 2, profvs);
	}
	
	delete[] profvs;
	
	for(int i=n-1;i>0;--i) {
		bld.connectProfiles(i, i-1, i >= halfN );
	}
	
	createMesh(bld);
}

}
