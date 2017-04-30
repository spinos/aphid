/*
 *  Primitive.h
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

namespace aphid {

class Primitive {

	int m_geomComp;
	
public:
	Primitive();

	void setGeometryComponent(const int & geom, const int & comp);
	void getGeometryComponent(int & geom, int & comp);
	
private:
	
};

}