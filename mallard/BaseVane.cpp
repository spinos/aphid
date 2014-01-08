/*
 *  BaseVane.cpp
 *  mallard
 *
 *  Created by jian zhang on 12/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 *
 *    V      L
 *    |      |
 *    c -----|-----
 *    |      |
 *    c -----|----- U
 */

#include "BaseVane.h"

BaseVane::BaseVane() : m_rails(0), m_gridU(0), m_gridV(0) {}

BaseVane::~BaseVane() 
{
    if(m_rails) delete[] m_rails;
}

void BaseVane::create(unsigned gridU, unsigned gridV)
{
     if(m_rails) delete[] m_rails;
     m_rails = new BezierCurve[gridV + 1];
     for(unsigned i=0; i <= gridV; i++)
         m_rails[i].createVertices(gridU + 1);
     m_gridU = gridU;
     m_gridV = gridV;
     m_profile.createVertices(gridV + 1);
}

BezierCurve * BaseVane::profile(unsigned idx) const
{
    return &m_rails[idx];
}

void BaseVane::computeKnots()
{
	for(unsigned i=0; i <= m_gridV; i++) m_rails[i].computeKnots();
}

void BaseVane::setU(float u)
{
	for(unsigned i=0; i <= m_gridV; i++) {
        m_profile.m_cvs[i] = m_rails[i].interpolate(u);
    }
    m_profile.computeKnots();
}

void BaseVane::pointOnVane(float v, Vector3F & dst)
{
    dst = m_profile.interpolate(v);
}

Vector3F * BaseVane::railCV(unsigned u, unsigned v)
{
	return &m_rails[v].m_cvs[u];
}

unsigned BaseVane::gridU() const
{
	return m_gridU;
}
	
unsigned BaseVane::gridV() const
{
	return m_gridV;
}

BezierCurve * BaseVane::profile()
{
	return &m_profile;
}

BezierCurve * BaseVane::rails()
{
	return m_rails;
}

void BaseVane::copy(BaseVane & another)
{
	const unsigned nu = another.gridU();
	const unsigned nv = another.gridV();
	create(nu, nv);
	for(unsigned i = 0; i <= nu; i++) {
		for(unsigned j = 0; j <=nv; j++) {
			*railCV(i, j) = *(another.railCV(i, j));
		}
	}	
	computeKnots();
}