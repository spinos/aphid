#pragma once

class DensityEstimator {
public:
    DensityEstimator();
    virtual ~DensityEstimator();
    
    void setSmoothRadius(const float & h);
    const float smoothRadius() const;
protected:
    const float weightPoly6(const float & r2) const;
    const float weightSpiky(const float & r) const;
    const float kernelPoly6() const;
    const float kernelSpiky() const;
 
private:
    float m_smoothRadius, m_coeffPoly6, m_coeffSpiky, m_h2;
};
