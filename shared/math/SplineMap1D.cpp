#include "SplineMap1D.h"

namespace aphid {

SplineMap1D::SplineMap1D() 
{
    m_spline.cv[0].set(0.f, 1.f, 0.f);
    m_spline.cv[1].set(.5f, 1.f, 0.f);
    m_spline.cv[2].set(.5f, .5f, 0.f);
    m_spline.cv[3].set(1.f, .5f, 0.f);
}

SplineMap1D::SplineMap1D(float a, float b)
{
    m_spline.cv[0].set(0.f, a, 0.f);
    m_spline.cv[1].set(.5f, a, 0.f);
    m_spline.cv[2].set(.5f, b, 0.f);
    m_spline.cv[3].set(1.f, b, 0.f);
}

SplineMap1D::~SplineMap1D()
{}

void SplineMap1D::setStart(float y)
{ m_spline.cv[0].y = y; }

void SplineMap1D::setEnd(float y)
{ m_spline.cv[3].y = y; }

void SplineMap1D::setLeftControl(float x, float y)
{
    m_spline.cv[1].x = x;
    m_spline.cv[1].y = y;
}

void SplineMap1D::setRightControl(float x, float y)
{
    m_spline.cv[2].x = x;
    m_spline.cv[2].y = y;
}

float SplineMap1D::interpolate(float t) const
{ return m_spline.calculateBezierPoint(t).y; }

BezierSpline * SplineMap1D::spline()
{ return &m_spline; }

}
//:~