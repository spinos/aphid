/*
 *  SplineCylinder.cpp
 *
 *  cylinder with spline adjust radius and height
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "SplineCylinder.h"
#include <math/miscfuncs.h>

namespace aphid {

SplineCylinder::SplineCylinder()
{}

SplineCylinder::~SplineCylinder()
{}

void SplineCylinder::createCylinder(int nu, int nv, float radius, float height)
{
	const float dv = 1.f / (float)nv;
	float mh = 0.f;
	float* hs = new float[nv+1];
	for(int i=0;i<nv;++i) {
		hs[i] = mh;
		
		float d = m_heightSpline.interpolate(dv * i);
		if(d < .01f) d = .01f;
		
		mh += d;
	}
	hs[nv] = mh;
	
/// scale back to height
	const float factor = height / mh;
	
	for(int i=0;i<=nv;++i) {
		hs[i] *= factor;
	}
	
	createCylinder1(nu, nv, radius, height, hs);
	adjustRadius();
	delete[] hs;
}

SplineMap1D* SplineCylinder::radiusSpline()
{ return &m_radiusSpline; }

SplineMap1D* SplineCylinder::heightSpline()
{ return &m_heightSpline; }

void SplineCylinder::adjustRadius()
{
	const float dh = 1.f / height();
	const int np = numPoints();
	
	for(int i=0;i<np;++i) {
		Vector3F& pv = points()[i];
		float d = m_radiusSpline.interpolate(pv.y * dh);
		if(d < .01f) d = .01f;
		
		pv.x *= d;
		pv.z *= d;
	}
	
}

}