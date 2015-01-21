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

void testBarycentric4()
{
    std::cout<<"\n test barycentric coordinate in tetrahedron\n";
    Vector3F tet[4];
	tet[0].set(0.f, -1.f, 0.f);
	tet[1].set(1.f, -1.f, 0.f);
	tet[2].set(0.f, -1.f, 1.f);
	tet[3].set(0.f, 1.f, 0.f);
	
	Vector3F test(0.f, .5f, 0.f);
	BarycentricCoordinate coord = getBarycentricCoordinate4(test, tet);
	
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

int main(int argc, char * const argv[])
{
	std::cout<<"GJK intersection test\n";
	testBarycentric3();
	testBarycentric4();
	std::cout<<"end of test\n";
	return 0;
}