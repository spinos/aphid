/*
 *  TetraDistance.cpp
 *  
 *
 *  Created by zhang on 17-2-4.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "TetraDistance.h"
#include <math/Plane.h>

namespace aphid {

namespace ttg {

TetraDistance::TetraDistance(const cvx::Tetrahedron & tet)
{
    const Vector3F cen = tet.getCenter();
    m_p[0] = cen + (tet.X(0) - cen) *.9f;
    m_p[1] = cen + (tet.X(1) - cen) *.9f;
    m_p[2] = cen + (tet.X(2) - cen) *.9f;
    m_p[3] = cen + (tet.X(3) - cen) *.9f;
}

TetraDistance::~TetraDistance()
{}

void TetraDistance::compute(const Plane & pl)
{
    m_d[0] = pl.distanceTo(m_p[0]);
    m_d[1] = pl.distanceTo(m_p[1]);
    m_d[2] = pl.distanceTo(m_p[2]);
    m_d[3] = pl.distanceTo(m_p[3]);
}

const float * TetraDistance::result() const
{ return m_d; }

}

}

