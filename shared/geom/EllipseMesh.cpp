/*
 *  EllipseMesh.cpp
 *  
 *
 *  Created by jian zhang on 8/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "EllipseMesh.h"

namespace aphid {

EllipseMesh::EllipseMesh()
{}

EllipseMesh::~EllipseMesh()
{}

void EllipseMesh::createEllipse(const float& width, const float& height,
					const int& m, const int& n)
{
	const float deltaX = width * .5f / (float)n;
	int realM = m;
	const int npf = n * 2 + 1;
	if(realM < npf + 1)
		realM = npf + 1;
		
	const int m1 = realM + 1;
		
	float* deltaYs = new float[m1];
	const float dy = 1.f / (float)realM;
	float sumy = 0.f;
	for(int i=0;i<realM;++i) {
		float d = heightSpline()->interpolate(dy * i);
		if(d < .09f)
			d = .09f;
			
		d *= dy;
		
		deltaYs[i] = d;
		sumy += d;
	}
	deltaYs[realM] = dy;
	
	const float yscale = height / sumy;
	for(int i=0;i<m1;++i) {
		deltaYs[i] *= yscale;
	}
	
	LoftMeshBuilder bld;
	
	int* rowbegins = new int[m1];
	
	Vector3F pv(0.f, 0.f, 0.f);
	float ldx = 0.f;
	float rdx = 0.f;
	int nvpr = 1;
	int tip = 0;
/// add points in rows
	for(int j=0;j<m1;++j) {
	
		rowbegins[j] = bld.numPoints();
			
		float zscale = pv.y / height;
		zscale = veinVarySpline()->interpolate(zscale);
		
		for(int i=0;i<nvpr;++i) {
			
			if(nvpr > 1) {
				int di = i - (nvpr>>1);
				if(di < 0)
					di = -di;
					
				float uparam = (float)di / (float)((nvpr>>1)); 
				float vz = veinSpline()->interpolate(uparam) - .5f;
				pv.z = vz * width * .789f * zscale * (float)nvpr / (float)npf;
				
					
			} else {
				pv.z = 0.f;
			}
			
			bld.addPoint(pv);
			
			if(i< (nvpr>>1) )
				pv.x += ldx;
			else
				pv.x += rdx;
			
		}
		
		if(j < n) {
			nvpr += 2;
			tip++;
		} else if(j > realM - n - 1) {
			nvpr -= 2;
			tip--;
		} else {
			nvpr = npf;
		}
/// advance y
		pv.y += deltaYs[j];
		
		ldx = leftSpline()->interpolate(pv.y / height );
		if(ldx < .07f)
			ldx = .07f;
		ldx *= deltaX;
			
		rdx = rightSpline()->interpolate(pv.y / height );
		if(rdx < .07f)
			rdx = .07f;
		rdx *= deltaX;
		
		if(tip > 0
			&& nvpr < npf) {
			float rat = (float)npf/(float)nvpr;
			ldx *= rat;
			rdx *= rat;
		}
		
/// move to left start x
		pv.x = -ldx * (nvpr >> 1);
	}
	
	const int halfNpf = npf>>1;
	
	int* profvs = new int[m1];
	for(int j=0;j<npf;++j) {
	
		nvpr = 1;
		tip = 0;
		for(int i=0;i<m1;++i) {
			
			profvs[i] = rowbegins[i] + j;
			
			if(nvpr < npf) {
				
				const int mergb = (nvpr - 1)>>1;
				if(j > mergb ) {
					profvs[i] = rowbegins[i] + mergb;	
				}
				
				if(j > npf - 1 - tip) {
					profvs[i] += j - npf + 1 + tip;
				}
			}
			
			if(i < n) {
				nvpr += 2;
				tip++;
			} else if(i > realM - n - 1) {
				nvpr -= 2;
				tip--;
			} else {
				nvpr = npf;
			}
			
		}
		
		bld.addProfile(m1, profvs);
	}
	
	delete[] profvs;
	delete[] rowbegins;
	delete[] deltaYs;
	
	for(int i=npf-1;i>0;--i) {
		bld.connectProfiles(i, i-1, i <= n );
	}
	
	createMesh(bld);
}

SplineMap1D* EllipseMesh::leftSpline()
{ return &m_leftSpline; }

SplineMap1D* EllipseMesh::rightSpline()
{ return &m_rightSpline; }

SplineMap1D* EllipseMesh::heightSpline()
{ return &m_heightSpline; }

SplineMap1D* EllipseMesh::veinSpline()
{ return &m_veinSpline; }

SplineMap1D* EllipseMesh::veinVarySpline()
{ return &m_veinVarySpline; }

}
