/*
 *  TetraDistance.cpp
 *  
 *
 *  Created by zhang on 17-2-4.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "TetraDistance.h"
#include <geom/ConvexShape.h>

namespace aphid {

namespace ttg {

TetraDistance::TetraDistance()
{}

TetraDistance::~TetraDistance()
{}

void TetraDistance::compute(const Plane & pl)
{
    for(int i=0;i<4;++i) {
        m_d[i] = pl.distanceTo(P(i));
        m_valid[i] = true;
    }
}

const float * TetraDistance::result() const
{ return m_d; }

const bool * TetraDistance::isValid() const
{ return m_valid; }

void TetraDistance::setDistance(const float & x, int i)
{ m_d[i] = x; }

void TetraDistance::setIndices(const int * x)
{
	memcpy(m_indices, x, 16);
}

const int & TetraDistance::snapInd() const
{ return m_snapInd; }
	
const Vector3F & TetraDistance::snapPos() const
{ return m_snapPos; }

}

}

