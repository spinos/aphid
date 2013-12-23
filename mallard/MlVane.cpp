/*
 *    V      L
 *    |      |
 *    c -----|-----
 *    |      |
 *    c -----|----- U
 *
 */

#include "MlVane.h"

MlVane::MlVane() : m_rails(0), m_gridU(0), m_gridV(0) {}

MlVane::~MlVane() 
{
    if(m_rails) delete[] m_rails;
}

void MlVane::create(unsigned gridU, unsigned gridV)
{
     if(m_rails) delete[] m_rails;
     m_rails = new BezierCurve[gridV + 1];
     for(unsigned i=0; i <= gridV; i++)
         m_rails[i].createVertices(gridU + 1);
     m_gridU = gridU;
     m_gridV = gridV;
     m_profile.createVertices(gridV + 1);
}

BezierCurve * MlVane::profile(unsigned idx) const
{
    return &m_rails[idx];
}

void MlVane::computeKnots()
{
	for(unsigned i=0; i <= m_gridV; i++)
         m_rails[i].computeKnots();
}

void MlVane::setU(float u)
{
	for(unsigned i=0; i <= m_gridV; i++) {
        m_profile.m_cvs[i] = m_rails[i].interpolate(u);
    }
    m_profile.computeKnots();
}

void MlVane::pointOnVane(float v, Vector3F & dst)
{
    dst = m_profile.interpolate(v);
}

Vector3F * MlVane::railCV(unsigned u, unsigned v)
{
	return &m_rails[v].m_cvs[u];
}

unsigned MlVane::gridU() const
{
	return m_gridU;
}
	
unsigned MlVane::gridV() const
{
	return m_gridV;
}
