/*
 *  AirfoilMesh.cpp
 *  
 *
 *  Created by jian zhang on 12/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "AirfoilMesh.h"
#include <cmath>

namespace aphid {

AirfoilMesh::AirfoilMesh(const float & c,
			const float & m,
			const float & p,
			const float & t) : Airfoil(c, m, p, t)
{}

AirfoilMesh::~AirfoilMesh()
{}

void AirfoilMesh::tessellate(const int & gx, const int & gy)
{
	unsigned nv = 2 + (gx - 1) * (gy * 2 + 1);
	unsigned nt = ((gx -2) * 4 + 4) * gy;
	
	create(nv, nt);
	
	Vector3F * pnt = points();
	unsigned * ind = indices();
	
	const float & c = chord();
	const float oonx = 1.f / (float)gx;
	const float oony = .5f / (float)gy;
	const float dx = c / (float)gx;
	float cb = camberRatio() / 0.1f;
	if(cb > 1.f) {
		cb = 1.f;
	}
	
	float vx, vy, yc, yt, theta, xu, xl, yu, yl, alpha;
	
	pnt[0].set(.001f, 0.f, 0.f);
	int it = 1;
	
/// row
	for(int i=1; i< gx;++i) {
		alpha = oonx * i;
		if(i>0) {
			alpha = alpha * alpha * (0.6f + 0.4 * (1.f - alpha)) + sqrt(alpha) * (0.4f * alpha);
		}
		
		vx = c * alpha;
		yc = calcYc(alpha);
		yt = calcYt(alpha);
		theta = calcTheta(vx);
		
		alpha = sin(theta);
		theta = cos(theta);
		xl = vx + yt * alpha;
		xu = vx - yt * alpha;
		yl = yc - yt * theta;
		yu = yc + yt * theta;
			
/// column
		for(int j = -gy; j<= gy;++j) {
		
			alpha = oony * (j+gy);
			alpha = alpha * (0.6f + 0.4f * (1.f - cb) ) + alpha * alpha * cb * 0.4f;
			
			Vector3F & pr = pnt[it];
			
			pr.x = xl + (xu - xl) * alpha;
			pr.y = yl + (yu - yl) * alpha;
			pr.z = 0.f;
			
			it++;
			
		}
	}
	
	pnt[it].set(c-.001f, 0.f, 0.f);
	
	const int last = it;
	
	unsigned * tri = indices();
	
	for(int i=0;i<nt*3;++i) {
		tri[i] = 0;
	}
	
	it = 0;
/// to first
	for(int i=0;i<gy * 2;++i) {
		tri[it++] = 0;
		tri[it++] = i+1;
		tri[it++] = i+2;
	}
	
	const int nvpc = 2*gy+1;
	
	for(int i=0; i< gx-2;++i) {
		
		for(int j = 0; j< gy*2;++j) {
		
			int lb = 1 + i*nvpc + j;
			
			tri[it++] = lb;
			tri[it++] = lb + nvpc;
			tri[it++] = lb + nvpc + 1;
			
			tri[it++] = lb;
			tri[it++] = lb + nvpc + 1;
			tri[it++] = lb + 1;
		}
	}
	
/// to last
	for(int i=0;i<gy * 2;++i) {
		tri[it++] = last;
		tri[it++] = last-i-1;
		tri[it++] = last-i-2;
	}
	
}

void AirfoilMesh::flipAlongChord()
{
	const float & c = chord();
	const int nv = numPoints();
	for(int i=0;i<nv;++i) {
		Vector3F & p = points()[i];
		p.x = c - p.x; 
	}
	
	const int nt = numTriangles();
	for(int i=0;i<nt;++i) {
		unsigned * tri = &indices()[i*3];
		unsigned swi = tri[0];
		tri[0] = tri[1];
		tri[1] = swi;
	}
}

}