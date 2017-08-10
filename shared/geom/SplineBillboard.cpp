/*
 *  SplineBillboard.cpp
 *
 *  billboard with spline adjust width
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "SplineBillboard.h"
#include <math/Vector2F.h>
#include <math/miscfuncs.h>

namespace aphid {

SplineBillboard::SplineBillboard(float w, float h) : BillboardMesh(w, h)
{}

SplineBillboard::~SplineBillboard()
{}

void SplineBillboard::setBillboardSize(float w, float h)
{
	BillboardMesh::setBillboardSize(w, h);
	adjustLeft();
	adjustRight();
}

SplineMap1D* SplineBillboard::leftSpline()
{ return &m_leftSpline; }

SplineMap1D* SplineBillboard::rightSpline()
{ return &m_rightSpline; }

void SplineBillboard::adjustLeft()
{
	const float hw = .5f * width();
	const float dv = 1.f / (float)nv();
	Vector3F * p = points();
	
	for(int i=0;i<=nv();++i) {
		float d = m_leftSpline.interpolate(dv * i);
		if(d < .05f) d = .05f;
		
		p[i * 2].x = -d * hw;
	}
	
	adjustTexcoord(false);
}

void SplineBillboard::adjustRight()
{
	const float hw = .5f * width();
	const float dv = 1.f / (float)nv();
	Vector3F * p = points();
	
	for(int i=0;i<=nv();++i) {
		float d = m_rightSpline.interpolate(dv * i);
		if(d < .05f) d = .05f;
		
		p[1 + i * 2].x = d * hw;
	}
	
	adjustTexcoord(true);
}

void SplineBillboard::adjustTexcoord(bool isLeft)
{
	float sv;
	if(widthHeightRatio() > 1.f) {
		sv = .995 / width();
	} else {
		sv = .995f / height();
	}
	float midu = width() * .5f * sv;
	
	Vector2F * texc = (Vector2F *)triangleTexcoords();
	const int nt = numTriangles();
	int acc=0;
	for(int i=0;i<nt;++i) {
		const int i3 = i * 3;
		for(int j=0;j<3;++j) {
			Vector2F& texcj = texc[i3 + j];
			adjustTexcoordPnt(texcj, midu, isLeft);
		}
	}
}

void SplineBillboard::adjustTexcoordPnt(Vector2F& texc,
			float midu, bool isLeft)
{
	if(isLeft) {
		if(texc.x < midu) {
			float d = m_leftSpline.interpolate(texc.y);
			Clamp01(d);
			
			texc.x = midu - d * midu;
		} 
	} else {
		if(texc.x > midu) {
			float d = m_rightSpline.interpolate(texc.y);
			Clamp01(d);
			
			texc.x = midu + d * midu;
		} 
	}
}

}