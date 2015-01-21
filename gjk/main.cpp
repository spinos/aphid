/*
 *  main.cpp
 *  
 *
 *  Created by jian zhang on 1/11/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <iostream>
#include <boost/format.hpp>
#include <Gjk.h>

void testBarycentric()
{
    Vector3F tet[4];
	tet[0].set(-1.f, .95f, 0.f);
	tet[1].set(0.f, 1.f, 2.f);
	tet[2].set(2.f, 1.f, 0.f);
	tet[3].set(0.f, 2.f, 0.f);
	
	Vector3F test(0.f, 1.5f, 0.f);
	BarycentricCoordinate coord = getBarycentricCoordinate(test, tet);
	
	std::cout<<"test "<<test.str()<<"\n";
    
    std::cout<<"coord "<<coord.x<<" "<<coord.y<<" "<<coord.z<<" "<<coord.w<<"\n";
    
	Simplex S;
	addToSimplex(S, tet[0]);
	addToSimplex(S, tet[1]);
	addToSimplex(S, tet[2]);
	addToSimplex(S, tet[3]);
	
	Vector3F cls = closestToOriginOnTetrahedron(S);
	std::cout<<"cloest test "<<cls.str()<<"\n";
}

void testGJK(const PointSet & A, const PointSet & B)
{
    int k = 0;
	Vector3F w;
	Vector3F v = A.X[0];
	
	Simplex W;
	
	for(int i=0; i < 99; i++) {
	    v.reverse();
	    w = A.supportPoint(v) - B.supportPoint(v.reversed());
	    
	    // std::cout<<" v"<<k<<" "<<v.str()<<"\n";	
	    // std::cout<<" w"<<k<<" "<<w.str()<<"\n";	
	    // std::cout<<" wTv "<<w.dot(v)<<"\n";
	    
	    if(w.dot(v) < 0.f) {
	        std::cout<<" minkowski difference contains the origin\n";
	        std::cout<<"separating axis ||v"<<k<<"|| "<<v.length()<<"\n";
	        break;
	    }
	    
	    addToSimplex(W, w);
	    
	    if(isOriginInsideSimplex(W)) {
	        std::cout<<" simplex W"<<k<<" contains origin, intersected\n";
	        break;
	    }
	    
	    // std::cout<<" W"<<k<<" d="<<W.d<<"\n";
	    
	    v = closestToOriginWithinSimplex(W);
	    
	    k++;
	}
}

int main(int argc, char * const argv[])
{
	std::cout<<"GJK intersection test\n";

	PointSet A, B;
	
	A.X[0].set(0.f, 0.f, 0.f);
	A.X[1].set(0.f, 0.f, 3.f);
	A.X[2].set(3.f, 0.f, 0.f);
	
	B.X[0].set(3.f, 0.f, 2.f);
	B.X[1].set(3.f, 3.f, 2.f);
	B.X[2].set(0.f, 3.f, 2.f);
	
	for(int i=0; i < 99; i++) {
	    B.X[0].y -= 0.034f;
	    B.X[1].y -= 0.034f;
	    B.X[2].y -= 0.034f;
	    std::cout<<" y "<<B.X[0].y<<"\n";
	    testGJK(A, B);
	}
	
	std::cout<<"end of test\n";
	return 0;
}