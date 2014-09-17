#include "DensityEstimator.h"
#include <cmath>
DensityEstimator::DensityEstimator()
{
    setSmoothRadius(1.f);
}

DensityEstimator::~DensityEstimator() {}

void DensityEstimator::setSmoothRadius(const float & h)
{
    m_smoothRadius = h;
    m_h2 = h * h;
    m_coeffPoly6 = 315.f / 64.f / 3.141592f / pow(h, 9);
    m_coeffSpiky = 45.f / 3.141592f / pow(h, 6);
}

const float DensityEstimator::smoothRadius() const
{
    return m_smoothRadius;
}

const float DensityEstimator::weightPoly6(const float & r2) const
{
    const float d = m_h2 - r2;
    return d * d * d;
}

const float DensityEstimator::weightSpiky(const float & r) const
{
    const float d = m_smoothRadius - r;
    return d * d;
}

const float DensityEstimator::kernelPoly6() const
{
    return m_coeffPoly6;
}

const float DensityEstimator::kernelSpiky() const
{
    return m_coeffSpiky;
}
