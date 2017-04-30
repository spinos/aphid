#pragma once
#include <AllMath.h>
class DensityEstimator {
public:
    DensityEstimator();
    virtual ~DensityEstimator();
    
    void setSmoothRadius(const float & h);
    const float smoothRadius() const;
protected:
    const float weightPoly6(const float & r2) const;
    const Vector3F weightSpiky(const Vector3F & pi, const Vector3F & pj) const;
    const float kernelPoly6() const;
    const float kernelSpiky() const;
 
private:
    float m_smoothRadius, m_coeffPoly6, m_coeffSpiky, m_h2;
};
