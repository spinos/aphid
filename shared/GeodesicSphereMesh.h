/*
 *  GeodesicSphereMesh.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseMesh.h>
namespace aphid {

class GeodesicSphereMesh : public BaseMesh {
public:
	GeodesicSphereMesh(unsigned level);
	virtual ~GeodesicSphereMesh();
	void setRadius(float r);
private:	
	void subdivide(unsigned level, unsigned & currentVertex, unsigned & currentIndex, Vector3F * p, unsigned * idx, Vector3F a, Vector3F b, Vector3F c, Vector3F d);
};

}