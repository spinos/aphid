/*
 *  FeatherMesh.cpp
 *  
 *
 *  Created by jian zhang on 1/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "FeatherMesh.h"
#include <math/Matrix44F.h>

using namespace aphid;

FeatherMesh::FeatherMesh(const float & c,
			const float & m,
			const float & p,
			const float & t) : AirfoilMesh(c, m, p, t)
{}

FeatherMesh::~FeatherMesh()
{}

void FeatherMesh::create(const int & gx,
				const int & gy)
{
	tessellate(gx, gy);
	flipAlongChord();
	
	Matrix44F rot;
	rot.setOrientations(Vector3F(0,0,-1),
						Vector3F(-1,0,0),
						Vector3F(0,1,0) );
			
	Vector3F * p = points();
	const int n = numPoints();
	for(int i=0;i<n;++i) {
		p[i] = rot.transform(p[i]);
	}
	
}