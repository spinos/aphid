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

char GjkContactSolver::pairContacted(const PointSet & A, const PointSet & B, ContactResult * result)
{
    int k = 0;
	Vector3F w;
	Vector3F v = A.X[0];
	if(v.length2() < TINY_VALUE) v = A.X[1];
	
	resetSimplex(m_W);
	for(int i=0; i < 99; i++) {
	    //v.reverse();
	    w = A.supportPoint(v.reversed()) - B.supportPoint(v);
	    
	    // std::cout<<" v"<<k<<" "<<v.str()<<"\n";	
	    // std::cout<<" w"<<k<<" "<<w.str()<<"\n";	
	    // std::cout<<" wTv "<<w.dot(v)<<"\n";
		// v is not close enough to v(A − B).
	    // http://www.bulletphysics.com/ftp/pub/test/physics/papers/jgt04raycast.pdf
	    if(v.length2() - w.dot(v) < 0.01f * v.length2()) {
			result->normal = v;
			result->point = w;
	        // std::cout<<" v is close to separate plane w.v\n";
	        // std::cout<<"separating axis ||v"<<k<<"|| "<<v.length()<<"\n";
	        return 0;
	    }
	    
	    addToSimplex(m_W, w);
 
	    if(isOriginInsideSimplex(m_W)) {
	        // std::cout<<" simplex W"<<k<<" contains origin, intersected\n";
	        result->normal = v;
			result->point = w;
			return 1;
	    }
	    
	    // std::cout<<" W"<<k<<" d="<<W.d<<"\n";
	    
	    v = closestToOriginWithinSimplex(m_W);

	    k++;
	}
	return 0;
}

const Simplex GjkContactSolver::W() const
{ return m_W; }