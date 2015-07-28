#include "BezierCurve.h"

class SplineMap1D {
public:
    SplineMap1D();
    SplineMap1D(float a, float b);
    virtual ~SplineMap1D();
    
    float interpolate(float t) const;
    
    void setStart(float y);
    void setEnd(float y);
    void setLeftControl(float x, float y);
    void setRightControl(float x, float y);
protected:

private:
    BezierSpline m_spline;
};
