#pragma once
#include <DensityEstimator.h>
#include <maya/MMatrixArray.h>
#include <maya/MIntArray.h>

class MDensityEstimator : public DensityEstimator {
public:
    MDensityEstimator();
    virtual ~MDensityEstimator();
    void calculate(const MMatrixArray & spaces, const MIntArray & indices);
    const float getDensity(const int & i) const;
private:
    float * m_dens;
};
