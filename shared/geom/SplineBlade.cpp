/*
 *  SplineBlade.cpp
 *
 *  blade with spline adjust width
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "SplineBlade.h"
#include <math/Vector2F.h>
#include <math/miscfuncs.h>

namespace aphid {

SplineBlade::SplineBlade()
{}

SplineBlade::~SplineBlade()
{}

void SplineBlade::createBlade(const float& width, const float& height,
					const float& ribWidth, const float& tipHeight,
					const int& m, const int& n)
{
	BladeMesh::createBlade(width, height, ribWidth, tipHeight,
					m, n);
	const float hw = width * .5f;
	const float rh = height - tipHeight;
	adjustProfiles(&m_leftSpline, m, 0, (n>>1), hw, rh);
	adjustProfiles(&m_rightSpline, m, (n>>1), n, hw, rh);
	
	BoundingBox bbx;
	LoftMeshBuilder bld;
	bld.projectTexcoord(this, bbx);
}

SplineMap1D* SplineBlade::leftSpline()
{ return &m_leftSpline; }

SplineMap1D* SplineBlade::rightSpline()
{ return &m_rightSpline; }

SplineMap1D* SplineBlade::veinSpline()
{ return &m_veinSpline; }

void SplineBlade::adjustProfiles(const SplineMap1D* spl,
		const int& m, const int& n0, const int& n1,
		const float& relWidth,
		const float& relHeight)
{
	Vector3F * p = points();
	
	int vbegin, vend;
	for(int j=n0;j<n1;++j) {
		getProfileRange(vbegin, vend, j);
		for(int i=vbegin;i<vend;++i) {
			
			Vector3F& pv = p[getProfileVertex(i)];
/// height based
			float d = spl->interpolate(pv.y / relHeight );
			if(d < .09f) d = .09f;
			
			float uparam = pv.x / relWidth;
			if(uparam < 0.f)
				uparam = -uparam;
			
			pv.x *= d;
		
			float d1 = m_veinSpline.interpolate(uparam ) - .5f;
			pv.z += relWidth * d1 * d * 1.4f;
			
		}
	}	
}

}