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
#include "./3d/SimpleSystem.h"
#include "GjkContactSolver.h"

void testBarycentric3()
{
    std::cout<<"\n test barycentric coordinate in triangle\n";
    Vector3F tri[3];
	tri[0].set(-1.f, -1.f, 0.f);
	tri[1].set(1.f, -1.f, 0.f);
	tri[2].set(0.f, -1.f, 1.f);
	
	Vector3F test(0.f, -1.f, .003f);
	BarycentricCoordinate coord = getBarycentricCoordinate3(test, tri);
	std::cout<<"test "<<test.str()<<"\n";
    
    std::cout<<"coord "<<coord.x<<" "<<coord.y<<" "<<coord.z<<"\n";  
}

void testRayCast()
{
    std::cout<<"\n test CSO ray cast\n";
    PointSet A, B;
    A.X[0].set(-2.f, -2.f, 0.1f);
	A.X[1].set(2.f, -2.f, 0.1f);
	A.X[2].set(-2.f, 2.f, 0.1f);
	
	B.X[0].set(-1.9f, 2.1f, 4.01f);
	B.X[1].set(2.1f, 2.2f, 4.01f);
	B.X[2].set(2.1f, -2.3f, 4.01f);
	
	GjkContactSolver gjk;
	ClosestTestContext result;
	result.referencePoint.setZero();
	result.needContributes = 1;
	result.distance = 1e9;
	result.hasResult = 0;
	resetSimplex(result.W);
	
	gjk.separateDistance(A, B, &result);
	
	if(result.hasResult) std::cout<<" contacted \n";
	else {
	    std::cout<<" not contacted \n";
	    std::cout<<" separating axis from B to A "<<result.closestPoint.str()<<"\n";
	}
	
	// direction of relative velocity
	Vector3F r(1.f, 0.f, -1.f); r.normalize();
	std::cout<<" r "<<r.str()<<"\n";
	// ray length
	float lamda = 0.f;
	// ray started at origin
	const Vector3F startP = Vector3F::Zero;
	Vector3F hitP = startP;
	Vector3F hitN; hitN.setZero();
	Vector3F v = hitP - result.closestPoint;
	Vector3F w, p, pa, pb, localA, localB;
	resetSimplex(result.W);

	float vdotw, vdotr;
	int k = 0;
	while(v.length2() > TINY_VALUE) {
	    std::cout<<" v"<<k<<" "<<v.str()<<" len "<<v.length()<<"\n";
	    vdotr = v.dot(r);
	    
	    // SA-B(v)
	    pa = A.supportPoint(v, result.transformA, localA);
		pb = B.supportPoint(v.reversed(), result.transformB, localB);
	    p = pa - pb;
	    std::cout<<" p "<<p.str()<<"\n";
	    w = hitP - p;
	    vdotw = v.dot(w); 
	    
	    std::cout<<" w"<<k<<" "<<w.str()<<" len "<<w.length()<<"\n";
	    std::cout<<" v.w "<<vdotw<<" v.r "<<vdotr<<"\n";
	    std::cout<<" v.w / w.r "<<vdotw / vdotr<<"\n";
	    
	    if(vdotr > 0.f) {
	        std::cout<<" v.r > 0 missed\n";
	        break;
	    }
	    
	    if(vdotw < TINY_VALUE) {
	        std::cout<<" v.w < 0 missed\n";
	        break;
	    }
	    
	    lamda -= vdotw / vdotr;
	    std::cout<<" lamda "<<lamda<<"\n";
	    hitP = startP + r * lamda;              
	    std::cout<<" hit p "<<hitP.str()<<"\n";
	    hitN = v;
	    
	    result.hasResult = 0;
	    result.distance = 1e9;
	    result.referencePoint = hitP;
	
	    // update normal
	    gjk.separateDistance(A, B, &result);
	    
	    std::cout<<"closest p "<<result.closestPoint.str()<<"\n";
	    v = hitP - result.closestPoint;
	    std::cout<<"||v|| "<<v.length()<<"\n";
	    k++;
	}
}

void testM()
{
    Matrix33F mat;
    mat.setIdentity();
    *mat.m(0,0) = 2.f;
    *mat.m(1,1) = 2.f;
    *mat.m(2,2) = 2.f;
    
    std::cout<<"M"<<mat.str();
    
    Matrix33F matInv = mat;
    matInv.inverse();
    std::cout<<"M-1"<<matInv.str();
    
    Matrix33F mmi = mat * matInv;
    std::cout<<"M M-1"<<mmi.str();
}

int main(int argc, char * const argv[])
{
	std::cout<<"GJK intersection test\n";
	testM();
	std::cout<<"end of test\n";
	return 0;
}