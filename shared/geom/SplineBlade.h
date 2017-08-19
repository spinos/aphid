/*
 *  SplineBlade.h
 *
 *  blade with spline adjust width
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_SPLINE_BLADE_H
#define APH_SPLINE_BLADE_H

#include "BladeMesh.h"
#include <math/SplineMap1D.h>

namespace aphid {
    
class SplineBlade : public BladeMesh {
     	
	SplineMap1D m_leftSpline;
	SplineMap1D m_rightSpline;
/// horizontal
	SplineMap1D m_veinSpline;
	
public:
	SplineBlade();
    virtual ~SplineBlade();
	
	virtual void createBlade(const float& width, const float& height,
					const float& ribWidth, const float& tipHeight,
					const int& m, const int& n);
	
	SplineMap1D* leftSpline();
	SplineMap1D* rightSpline();
	SplineMap1D* veinSpline();
	
protected:
    void adjustProfiles(const SplineMap1D* spl,
		const int& m, const int& n0, const int& n1,
		const float& relWidth,
		const float& relHeight);
			
private:
    
};

}
#endif
