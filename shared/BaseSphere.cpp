#include "BaseSphere.h"

BaseSphere::BaseSphere() 
{
    m_radius = 0.f;
}

BaseSphere::~BaseSphere(){}

void BaseSphere::setCenter(const Vector3F & pos)
{
    m_center = pos;
}

void BaseSphere::setRadius(float r)
{
    m_radius = r;
}

Vector3F BaseSphere::center() const
{
    return m_center;
}

float BaseSphere::radius() const
{
    return m_radius;
}

char BaseSphere::inside(const BaseSphere & another, float & delta) const
{
    delta = Vector3F(m_center, another.center()).length() - m_radius;
    return delta > 0.f;
}

char BaseSphere::expand(const BaseSphere & another)
{
    float d;
    if(another.inside(*this, d)) return 0;
    m_radius += d;
    return 1;
}
