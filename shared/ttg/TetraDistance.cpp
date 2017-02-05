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

TetraDistance::TetraDistance(const cvx::Tetrahedron & tet)
{
    m_maxDistance = 0.f;
    const Vector3F cen = tet.getCenter();
    for(int i=0;i<4;++i) {
        Vector3F dv = tet.X(i) - cen;
        m_maxDistance += dv.length();
        m_p[i] = cen + dv *.8f;
        m_valid[i] = false;
    }
    m_maxDistance *= .473f;
}

TetraDistance::~TetraDistance()
{}

void TetraDistance::compute(const Plane & pl)
{
    for(int i=0;i<4;++i) {
        m_d[i] = pl.distanceTo(m_p[i]);
        m_valid[i] = true;
    }
}

const float * TetraDistance::result() const
{ return m_d; }

const bool * TetraDistance::isValid() const
{ return m_valid; }

}

}

