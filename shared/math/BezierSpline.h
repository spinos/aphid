/*
 *  bezierSpline.h
 *  
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_BEZIER_SPLINE_H
#define APH_BEZIER_SPLINE_H

#include "BoundingBox.h"

namespace aphid {

struct BezierSpline {

    void deCasteljauSplit(BezierSpline & a, BezierSpline & b)
    {
        Vector3F d = cv[1] * .5f + cv[2] * .5f;
        a.cv[0] = cv[0];
        a.cv[1] = cv[0] * .5f + cv[1] * .5f;
        a.cv[2] = a.cv[1] * .5f + d * .5f; 
        
        b.cv[3] = cv[3];
        b.cv[2] = cv[3] * .5f + cv[2] * .5f;
        b.cv[1] = b.cv[2] * .5f + d * .5f;
        
        a.cv[3] = b.cv[0] = a.cv[2] * .5f + b.cv[1] * .5f;
    }
    
    Vector3F calculateBezierPoint(float t) const
    {
        float u = 1.f - t;
        float tt = t * t;
        float uu = u*u;
        float uuu = uu * u;
        float ttt = tt * t;
        
        Vector3F p = cv[0] * uuu; //first term
        p += cv[1] * 3.f * uu * t; //second term
        p += cv[2] * 3.f * u * tt; //third term
        p += cv[3] * ttt; //fourth term
        return p;
    }
    
    bool straightEnough() const
    {
        const float d = cv[0].distanceTo(cv[3]);
        return ( ( cv[0].distanceTo(cv[1]) + cv[1].distanceTo(cv[2]) + cv[2].distanceTo(cv[3]) - d ) / d ) < .0001f;
    }
	
	void getAabb(BoundingBox * box) const
	{
		box->expandBy(cv[0]);
		box->expandBy(cv[1]);
		box->expandBy(cv[2]);
		box->expandBy(cv[3]);
	}
    
    Vector3F cv[4];
};

}

#endif
