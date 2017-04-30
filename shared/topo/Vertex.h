/*
 *  Vertex.h
 *  convexHull
 *
 *  Created by jian zhang on 9/5/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <topo/GeoElement.h>

namespace aphid {

class Vector3F;

class Vertex : public GeoElement {
public:
	Vertex();
	virtual ~Vertex();
	
	Vector3F *m_v;
};

}