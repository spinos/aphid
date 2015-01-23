/*
 *  GjkContactSolver.cpp
 *  proof
 *
 *  Created by jian zhang on 1/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "GjkContactSolver.h"

GjkContactSolver::GjkContactSolver() {}

void GjkContactSolver::distance(const PointSet & A, const PointSet & B, ClosestTestContext * result)
{
    int k = 0;
	float v2;
	Vector3F w;
	Vector3F v = A.X[0];
	if(v.length2() < TINY_VALUE) v = A.X[1];
	
	resetSimplex(m_W);
	for(int i=0; i < 99; i++) {
	    // SA-B(-v)
		w = supportMapping(A, B, v.reversed());
	    
		// terminate when v is close enough to v(A - B).
	    // http://www.bulletphysics.com/ftp/pub/test/physics/papers/jgt04raycast.pdf
		v2 = v.length2();
	    if(v2 - w.dot(v) < 0.01f * v2) {
	        std::cout<<" v is close to w "<<v2 - w.dot(v)<<"\n";
			result->hasResult = 0;
			break;
	    }
	    
	    addToSimplex(m_W, w);
 
	    if(isPointInsideSimplex(m_W, result->referencePoint)) {
	        std::cout<<" Minkowski difference contains the reference point\n";
			result->hasResult = 1;
			break;
	    }
	    
	    closestOnSimplex(m_W, result);
	    v = result->resultPoint - result->referencePoint;
	    k++;
	}
}

const Simplex GjkContactSolver::W() const
{ return m_W; }