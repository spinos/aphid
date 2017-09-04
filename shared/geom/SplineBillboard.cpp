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
#include "PlanarTexcoordProjector.h"
#include <math/miscfuncs.h>

namespace aphid {

SplineBillboard::SplineBillboard()
{
	m_centerSpline.setStart(.5f);
	m_centerSpline.setEnd(.5f);
	m_centerSpline.setLeftControl(.4f, .5f);
	m_centerSpline.setRightControl(.6f, .5f);
}

SplineBillboard::~SplineBillboard()
{}

void SplineBillboard::setBillboardSize(float w, float h, int nu, int addNv)
{
	BillboardMesh::setBillboardSize(w, h, nu, addNv);
	adjustCenter();
	adjustLeft();
	adjustRight();
	PlanarTexcoordProjector proj;
	BoundingBox bbx;
	proj.projectTexcoord(this, bbx);
}

SplineMap1D* SplineBillboard::centerSpline()
{ return &m_centerSpline; }

SplineMap1D* SplineBillboard::leftSpline()
{ return &m_leftSpline; }

SplineMap1D* SplineBillboard::rightSpline()
{ return &m_rightSpline; }

void SplineBillboard::adjustCenter()
{ 
	if(nu() & 1) 
		return;
		
	const float dv = 1.f / (float)nv();
	Vector3F * p = points();
	
	int ci = nu() >> 1;
	
	const int stripe = nu() + 1;
	
	for(int i=0;i<=nv();++i) {
		float d = m_centerSpline.interpolate(dv * i) - .5f;
		
		p[stripe * i + ci].x = d * width();
	}

}

void SplineBillboard::adjustLeft()
{
	const float hw = .5f * width();
	const float dv = 1.f / (float)nv();
	Vector3F * p = points();
	
	const int stripe = nu() + 1;
	
	for(int i=0;i<=nv();++i) {
		float d = m_leftSpline.interpolate(dv * i);
		if(d < .07f) d = .07f;
		
		p[stripe * i].x = rowCenter(i) - d * hw;
	}
	
}

void SplineBillboard::adjustRight()
{
	const float hw = .5f * width();
	const float dv = 1.f / (float)nv();
	Vector3F * p = points();
	
	const int stripe = nu() + 1;
	
	for(int i=0;i<=nv();++i) {
		float d = m_rightSpline.interpolate(dv * i);
		if(d < .07f) d = .07f;
		
		p[stripe * i + nu()].x = rowCenter(i) + d * hw;
	}
	
}

float SplineBillboard::rowCenter(int i) const
{
	if(nu() < 2) 
		return 0.f;
		
	return points()[i * (nu()+1) + (nu() >> 1)].x;
}

}