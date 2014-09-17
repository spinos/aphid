#include "MDensityEstimator.h"
#include <maya/MGlobal.h>
#include <maya/MVector.h>
MDensityEstimator::MDensityEstimator() : m_dens(0) {}
MDensityEstimator::~MDensityEstimator() 
{
    if(m_dens) delete[] m_dens;
}

void MDensityEstimator::calculate(const MMatrixArray & spaces, const MIntArray & indices)
{
    const int count = indices.length();
    if(count < 2) return;
    
    if(m_dens) delete[] m_dens;
    m_dens = new float[count];
    
    int i, j;
    float sum, r;
    MVector p, q;
    MMatrix m;
    for(i=0; i < count; i++) {
        m = spaces[indices[i]];
        p.x = m(3, 0);
        p.y = m(3, 1);
        p.z = m(3, 2);
        sum = 0.f;
        for(j=0; j < count; j++) {
            if(i == j) continue;
             m = spaces[indices[j]];
             q.x = m(3, 0);
             q.y = m(3, 1);
             q.z = m(3, 2);
             r = (p - q).length();
             if(smoothRadius() > r) sum += weightPoly6(r * r);
        }
        
        sum *= kernelPoly6();
        m_dens[i] = sum;
    }
}

const float MDensityEstimator::getDensity(const int & i) const
{
    return m_dens[i];    
}

