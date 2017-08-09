#ifndef APH_SPLINE_MAP_1D_H
#define APH_SPLINE_MAP_1D_H

#include "BezierSpline.h"

namespace aphid {

class SplineMap1D {

	 BezierSpline m_spline;
	 
public:
    SplineMap1D();
    SplineMap1D(float a, float b);
    virtual ~SplineMap1D();
/// t between [0,1]
    float interpolate(float t) const;
    
    void setStart(float y);
    void setEnd(float y);
    void setLeftControl(float x, float y);
    void setRightControl(float x, float y);
	BezierSpline * spline();
	
protected:

private:
   
};

}
#endif
