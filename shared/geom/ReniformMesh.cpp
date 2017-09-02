/*
 *  ReniformMesh.cpp
 *  
 *
 *  Created by jian zhang on 8/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ReniformMesh.h"
#include <math/miscfuncs.h>

namespace aphid {

ReniformMesh::ReniformMesh()
{}

ReniformMesh::~ReniformMesh()
{}

void ReniformMesh::createReniform(const ReniformMeshProfile& prof)
{
	const float hKidneyAng = .5f * prof._kidneyAngle;
	const float hInnerAng = .789f * hKidneyAng;
/// 6 inner segments
	const float dInnerAng = hInnerAng * .333333f;
	const float hStalkW = .5f * prof._stalkWidth;
	const float deltaStalkY = prof._stalkHeight / (float)prof._stalkSegments;
	const int midribM = 3;
	const float deltaMidridY = prof._midribHeight / (float)midribM;
/// 12 vein segments	
	static const float sVeinAngleScal[13] = {
	.2105f, .1842f, .1711f, .1579f, .1447f, .1316f,
	.1316f, .1447f, .1579f, .1711f, .1842f, .2105f,
	.1f,
	};
	
	LoftMeshBuilder bld;
	
	int nrows = 0;
	Vector3F pv(0.f, 0.f, 0.f);
/// add stalk and midrib points
	for(int i=0;i<prof._stalkSegments;++i) {
		pv.x = -hStalkW;
		bld.addPoint(pv);
		
		pv.x = hStalkW;
		bld.addPoint(pv);
		
		pv.y += deltaStalkY;
		nrows++;
	}
	
	int midribInds[7];
	int imidrib = 0;
	
	float midribScal = .9f;
	for(int i=0;i<midribM-1;++i) {
		pv.x = -hStalkW * midribScal;
		midribInds[imidrib++] = bld.numPoints();
		bld.addPoint(pv);
		
		pv.x = hStalkW * midribScal;
		midribInds[imidrib++] = bld.numPoints();
		bld.addPoint(pv);
		
		midribScal *= midribScal;
		pv.y += deltaMidridY;
		nrows++;
	}
	
	pv.x = -hStalkW * midribScal;
	midribInds[imidrib++] = bld.numPoints();
	bld.addPoint(pv);
		
	pv.x = hStalkW * midribScal;
	midribInds[imidrib++] = bld.numPoints();
	bld.addPoint(pv);
	nrows++;
	
	pv.x = 0.f;
	pv.y += deltaMidridY * midribScal;
	midribInds[imidrib++] = bld.numPoints();
	bld.addPoint(pv);
	nrows++;
	
	static const int midribIs[7] = {
		1, 3, 5, 
		6,
		4, 2, 0,
	};
	
/// inner points
	const int innerBegin = bld.numPoints();
	float curInnerAng = HALFPIF - hInnerAng;
	Vector3F bv;
	for(int i=0;i<7;++i) {
		Matrix33F rotm;
		rotm.rotateZ(curInnerAng);
		pv = rotm.transform(Vector3F::XAxis);
		
		pv *= prof._radius * .11f;
		
		int v0 = midribInds[midribIs[i]];
		bld.getPoint(bv, v0);
		bv += pv;
		bld.addPoint(bv);
		
		curInnerAng += dInnerAng;
	} 

	static const int innerIs[13] = {
		0, 0, 1, 1, 2, 2, 
		3,
		4, 4, 5, 5, 6, 6,
	};
	
/// vein points
	float deltaAscend = prof._ascendAngle / (float)prof._veinSegments;
	if(deltaAscend > 1.5f)
		deltaAscend = 1.5f;
	const float dveinL = prof._radius * .89f / (float)prof._veinSegments;
	const int veinBegin = bld.numPoints();
	Vector3F p0, p1;
	float curVeinAng = HALFPIF - hKidneyAng;
	for(int i=0;i<13;++i) {
		const float uparam = (float)i/12.f;
		float radialScale = 1.f;
		if(i<6)
			radialScale = rightSideSpline()->interpolate(uparam * 2.f);
		else
			radialScale = leftSideSpline()->interpolate((1.f - uparam ) * 2.f);
		if(radialScale < .05f)
			radialScale = .05f;
		float zScale = veinVarySpline()->interpolate(uparam) * .43f;
		if(zScale < .0f)
			zScale = .0f;
		
		Matrix33F rotm;
		rotm.rotateZ(curVeinAng);
		pv = rotm.transform(Vector3F::XAxis);		
		
		int v0 = innerBegin + innerIs[i];
		bld.getPoint(bv, v0);
		
		p0 = bv;
		for(int j=0;j<prof._veinSegments;++j) {
			
			float vparam = .99f;
			if(prof._veinSegments > 1)
				vparam =(float)j/(float)(prof._veinSegments - 1);
			
			float stretch = 1.f;
			if(i < 6) {
				if(pv.x > 1e-2f) {
					float ang = .5f * acos(pv.y);
					if(ang > deltaAscend * j)
						ang = deltaAscend * j;
						
					stretch = 1.f / cos(ang);
					rotm.setIdentity();
					rotm.rotateZ(ang);
					pv = rotm.transform(pv);
				}
			} else if(i > 6) {
				if(pv.x < -1e-2f) {
					float ang = .5f * acos(pv.y);
					if(ang > deltaAscend * j)
						ang = deltaAscend * j;
					
					stretch = 1.f / cos(ang);
					rotm.setIdentity();
					rotm.rotateZ(-ang);
					pv = rotm.transform(pv);
				}
			}
			
			p1 = p0 + pv * dveinL * radialScale * stretch;
			p1.z = prof._radius * zScale * (veinSpline()->interpolate(vparam) - .5f);
			
			bld.addPoint(p1 );
			
			p0 = p1;
			
		}
		
		curVeinAng += hKidneyAng * sVeinAngleScal[i];
	}

/// profile stalk and midrib
	int* profvs = new int[nrows];
	for(int j=0;j<2;++j) {
	
		for(int i=0;i<nrows-1;++i) {
			
			profvs[i] = (i<<1) + j;			
		}
		profvs[nrows - 1] = innerBegin - 1;
		
		bld.addProfile(nrows, profvs);
	}

/// profile vein
	for(int j=0;j<13;++j) {
	
		int v0 = innerIs[j];
		v0 = midribInds[midribIs[v0]];
		profvs[0] = v0;
		
		for(int i=1;i<2;++i) {
			
			profvs[i] = innerBegin + innerIs[j];			
		}
		
		for(int i=0;i<prof._veinSegments;++i)
			profvs[2 + i] = veinBegin + j * prof._veinSegments + i;
		
		bld.addProfile(2 + prof._veinSegments, profvs);
	}
	
	delete[] profvs;
	
	bld.connectProfiles(1, 0, true );
	
	for(int i=2;i<14;++i) 
		bld.connectProfiles(i, i+1, (i-2)>5 );
	
	createMesh(bld);
}

SplineMap1D* ReniformMesh::leftSideSpline()
{ return &m_leftSideSpline; }

SplineMap1D* ReniformMesh::rightSideSpline()
{ return &m_rightSideSpline; }

SplineMap1D* ReniformMesh::veinSpline()
{ return &m_veinSpline; }

SplineMap1D* ReniformMesh::veinVarySpline()
{ return &m_veinVarySpline; }

}
