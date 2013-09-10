/*
 *  CircleCurve.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "CircleCurve.h"
#include <AllMath.h>
CircleCurve::CircleCurve()
{   
}

void CircleCurve::create()
{
    m_numVertices = 37;
    m_cvs = new Vector3F[numVertices()];
    const float delta = PI / 18.f;
    for(int i = 0; i <= 36; i++)
		m_cvs[i].set(sin(delta * i), cos(delta * i), 0.f);
	
	computeKnots();
}

CircleCurve::~CircleCurve() {}
