/*
 *  Primitive.cpp
 *  kdtree
 *
 *  max 2^10-1 geometries
 *  max 2^22-1 components 
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "Primitive.h"

namespace aphid {

Primitive::Primitive() {}

void Primitive::setGeometryComponent(const int & geom, const int & comp)
{
	m_geomComp = ((geom<<22) | comp);
}

void Primitive::getGeometryComponent(int & geom, int & comp)
{
	geom = m_geomComp>>22;
	comp = (m_geomComp << 10)>>10;
}

}