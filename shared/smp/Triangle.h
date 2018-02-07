/*
 *  Triangle.h
 *  
 *  randomly sample point on triangle (A,B,C) input (r1,r2) [0,1]
 *  P = (1 - sqrt(r1)) * A + (sqrt(r1) * (1 - r2)) * B + (sqrt(r1) * r2) * C
 *  based on Shape Distributions by ROBERT OSADA et al.
 *
 *  Created by jian zhang on 2/10/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SMP_TRIANGLE_H
#define APH_SMP_TRIANGLE_H

#include <geom/ConvexShape.h>
#include <math/haltonSequence.h>
#include <math/miscfuncs.h>

namespace aphid {

namespace smp {

class Triangle {

	float m_coord[3];
/// area of sample
	float m_area;
	
public:

	Triangle();
	
	void setSampleSize(float x);
	
	int getNumSamples(const float& tarea);
	
/// T sample value
/// Tf interpolater
/// Triangle g
	template<typename T, typename Tf>
	bool sampleTriangle(T& samp, Tf& finterp, const cvx::Triangle* g);
	
};

template<typename T, typename Tf>
bool Triangle::sampleTriangle(T& samp, Tf& finterp, const cvx::Triangle* g)
{
	const float* r1r2 = Halton32Sequence256[rand() & 255];

	float r1 = r1r2[0];
	float r2 = r1r2[1];
	
	r1 = sqrt(r1);
	
	m_coord[0] = 1.f - r1;
	m_coord[1] = r1 * (1.f - r2);
	m_coord[2] = r1 * r2;
	
	samp._pos = g->P(0) * m_coord[0] 
				+ g->P(1) * m_coord[1] 
				+ g->P(2) * m_coord[2];
	
	if(finterp.reject(samp) )
		return false;
		
	finterp.interpolate(samp, m_coord, g);
	return true;
}

}

}

#endif