/*
 *  SplineCylinder.h
 *
 *  cylinder with spline adjust radius and height
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_SPLINE_CYLINDER_H
#define APH_SPLINE_CYLINDER_H

#include "CylinderMesh.h"
#include <math/SplineMap1D.h>

namespace aphid {
    
class SplineCylinder : public CylinderMesh {
     	
	SplineMap1D m_radiusSpline;
	SplineMap1D m_heightSpline;
	
public:
	SplineCylinder();
	virtual ~SplineCylinder();
	
	virtual void createCylinder(int nu, int nv, float radius, float height);
	
	SplineMap1D* radiusSpline();
	SplineMap1D* heightSpline();
	
	void adjustRadius();
	
protected:
    		
private:
    
};

}
#endif
