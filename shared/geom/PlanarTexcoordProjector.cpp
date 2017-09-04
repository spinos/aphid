/*
 *  PlanarTexcoordProjector.cpp
 *  
 *
 *  Created by jian zhang on 9/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "PlanarTexcoordProjector.h"
#include "ATriangleMesh.h"

namespace aphid {

PlanarTexcoordProjector::PlanarTexcoordProjector()
{ m_texcori = tCenteredZero; }

void PlanarTexcoordProjector::setTexcoordOrigin(PlanarTexcoordProjector::TexcoordOrigin x)
{ m_texcori = x; }

void PlanarTexcoordProjector::projectTexcoord(ATriangleMesh* msh,
						BoundingBox& bbx) const
{
	bbx = msh->calculateGeomBBox();
	const float width = bbx.distance(0);
	const float height = bbx.distance(1);
	const float woverh = width / height;
	float sr;
	if(woverh > 1.f) {
		sr = .995f / width;
	} else {
		sr = .995f / height;
	}
	
	float xcenter = .5f;
	float ycenter = .0025f;
	float xoffset = 0.f;
	float yoffset = 0.f;
	
	if(m_texcori == tLeftBottom) {
		const Vector3F offset = bbx.getMin();
		xoffset = -offset.x;
		yoffset = -offset.y;
		xcenter = .0025f;
		ycenter = .0025f;
	} else if(m_texcori == tCenteredBox) {
		xoffset = -.5f * bbx.distance(0);
	}
	
	float * texc = msh->triangleTexcoords();
	const int nt = msh->numTriangles();
	Vector3F * p = msh->points();
	unsigned * ind = msh->indices();
	
	int acc=0;
	for(int i=0;i<nt;++i) {
		const int i3 = i * 3;
		for(int j=0;j<3;++j) {
			
			const Vector3F& pj = p[ind[i3 + j] ];
			texc[acc++] = (pj.x + xoffset) * sr + xcenter;
			texc[acc++] = (pj.y + yoffset) * sr + ycenter;
		}
	}
}

}
