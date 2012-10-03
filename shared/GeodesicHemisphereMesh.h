/*
 *  GeodesicHemisphereMesh.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseMesh.h>

class GeodesicHemisphereMesh : public BaseMesh {
public:
	GeodesicHemisphereMesh(unsigned level);
	virtual ~GeodesicHemisphereMesh();
	
	void subdivide(unsigned level, unsigned & currentVertex, unsigned & currentIndex, Vector3F * p, unsigned * idx, Vector3F a, Vector3F b, Vector3F c, Vector3F d);
};