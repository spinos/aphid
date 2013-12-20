/*
 *    V      L
 *    |      |
 *    c -----|-----
 *    |      |
 *    c -----|----- U
 *
 */

#include "MlVane.h"

MlVane::MlVane() : m_rails(0) {}

MlVane::~MlVane() 
{
    if(m_rails) delete[] m_rails;
}

void MlVane::create(unsigned gridU, unsigned gridV)
{
     if(m_rails) delete[] m_rails;
     m_rails = new BezierCurve[gridV + 1];
     for(unsigned i=0; i < gridV; i++)
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

void MlVane::pointOnVane(float u, float v, Vector3F & dst)
{
    for(unsigned i=0; i <= m_gridV; i++) {
        m_profile.m_cvs[i] = m_rails[i].interpolate(u);
    }
    m_profile.computeKnots();
    dst = m_profile.interpolate(v);
}

