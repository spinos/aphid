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
	
	for(int i=0; i < 99; i++) {
	    // SA-B(-v)
		w = supportMapping(A, B, v.reversed());
	    
		// terminate when v is close enough to v(A - B).
	    // http://www.bulletphysics.com/ftp/pub/test/physics/papers/jgt04raycast.pdf
		v2 = v.length2();
	    if(v2 - w.dot(v) < 0.0001f * v2) {
	        // std::cout<<" v is close to w "<<v2 - w.dot(v)<<"\n";
			result->hasResult = 0;
			break;
	    }
	    
	    addToSimplex(result->W, w);
 
	    if(isPointInsideSimplex(result->W, result->referencePoint)) {
	        // std::cout<<" Minkowski difference contains the reference point\n";
			result->hasResult = 1;
			break;
	    }
	    
	    closestOnSimplex(result->W, result);
	    v = result->resultPoint - result->referencePoint;
	    k++;
	}
}

void GjkContactSolver::rayCast(const PointSet & A, const PointSet & B, ClosestTestContext * result)
{
	distance(A, B, result);
	if(result->hasResult) return;
	
	resetSimplex(result->W);
	
	const Vector3F r = result->rayDirection;
	float lamda = 0.f;
	// ray started at origin
	const Vector3F startP = Vector3F::Zero;
	Vector3F hitP = startP;
	Vector3F hitN; hitN.setZero();
	Vector3F v = hitP - result->resultPoint;
	Vector3F w, p;
	
	float vdotw, vdotr;
	int k = 0;
	for(; k < 99; k++) {
	    vdotr = v.dot(r);
	    
	    // SA-B(v)
	    p = supportMapping(A, B, v);
	    
		w = hitP - p;
	    vdotw = v.dot(w); 
	    
	    if(vdotw > 0.f) {
			// std::cout<<" v.w > 0\n";
			if(vdotr >= 0.f) {
				// std::cout<<" v.r >= 0 missed\n";
				result->hasResult = 0;
				return;
			}
			lamda -= vdotw / vdotr;
			hitP = startP + r * lamda;
			hitN = v;
		}
		
	    addToSimplex(result->W, p);
	    
	    result->hasResult = 0;
	    result->distance = 1e9;
	    result->referencePoint = hitP;
	    
	    closestOnSimplex(result->W, result);
	    
	    v = hitP - result->resultPoint;
		if(v.length2() < TINY_VALUE) break; 
	}
	
	if(k==99) std::cout<<"    max iterations reached!\n";
	std::cout<<"||v|| "<<k<<" "<<v.length()<<"\n";
	result->hasResult = 1;
	result->resultPoint = hitN;
	result->distance = lamda;
}
