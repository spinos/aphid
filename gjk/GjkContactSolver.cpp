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

char GjkContactSolver::pairContacted(const PointSet & A, const PointSet & B)
{
    int k = 0;
	Vector3F w;
	Vector3F v = A.X[0];
	if(v.length2() < TINY_VALUE) v = A.X[1];
	
	char contacted = 0;
	Simplex W;
	for(int i=0; i < 99; i++) {
	    v.reverse();
	    w = A.supportPoint(v) - B.supportPoint(v.reversed());
	    
	    // std::cout<<" v"<<k<<" "<<v.str()<<"\n";	
	    // std::cout<<" w"<<k<<" "<<w.str()<<"\n";	
	    // std::cout<<" wTv "<<w.dot(v)<<"\n";
	    
	    if(w.dot(v) < 0.f) {
	        // std::cout<<" minkowski difference contains the origin\n";
	        // std::cout<<"separating axis ||v"<<k<<"|| "<<v.length()<<"\n";
			//glColor3f(1.f, 0.f, 0.f);
            //glBegin(GL_LINES);
            //glVertex3f(0.f, 0.f, 0.f);
            //glVertex3f(v.x, v.y, v.z);
            //glEnd();
	        return 0;
	    }
	    
	    addToSimplex(W, w);
 
	    if(isOriginInsideSimplex(W)) {
	        // std::cout<<" simplex W"<<k<<" contains origin, intersected\n";
	        contacted = 1;
			// drawSimplex(W);
	        return 1;
	    }
	    
	    // std::cout<<" W"<<k<<" d="<<W.d<<"\n";
	    
	    v = closestToOriginWithinSimplex(W);
		
		if(v.length2() < TINY_VALUE) {
			contacted = 1;
			return 1;
		}

	    k++;
	    /*
	    if(k == m_drawLevel) {
	        drawSimplex(W);
	        glColor3f(1.f, 0.f, 0.f);
            glBegin(GL_LINES);
            glVertex3f(0.f, 0.f, 0.f);
            glVertex3f(v.x, v.y, v.z);
            glEnd();
			
			if(W.d == 2) drawLine(W.p[0], W.p[1]);
	    }*/
	}
	return 0;
}