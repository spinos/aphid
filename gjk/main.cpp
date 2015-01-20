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

#include <AllMath.h>

struct BarycentricCoordinate {
    float x, y, z, w;
};

float determinantTetrahedron(Matrix44F & mat, const Vector3F & v1, const Vector3F & v2, const Vector3F & v3, const Vector3F & v4)
{
    * mat.m(0, 0) = v1.x;
    * mat.m(0, 1) = v1.y;
    * mat.m(0, 2) = v1.z;
    * mat.m(0, 3) = 1.f;
    
    * mat.m(1, 0) = v2.x;
    * mat.m(1, 1) = v2.y;
    * mat.m(1, 2) = v2.z;
    * mat.m(1, 3) = 1.f;
    
    * mat.m(2, 0) = v3.x;
    * mat.m(2, 1) = v3.y;
    * mat.m(2, 2) = v3.z;
    * mat.m(2, 3) = 1.f;
    
    * mat.m(3, 0) = v4.x;
    * mat.m(3, 1) = v4.y;
    * mat.m(3, 2) = v4.z;
    * mat.m(3, 3) = 1.f;
    
    return mat.determinant();
}

BarycentricCoordinate getBarycentricCoordinate(const Vector3F & p, const Vector3F * v)
{
    Matrix44F mat;
    
    BarycentricCoordinate coord;
    
    float D0 = determinantTetrahedron(mat, v[0], v[1], v[2], v[3]);
    if(D0 == 0.f) {
        std::cout<<"tetrahedron is degenerate\n";
        coord.x = coord.y = coord.z = coord.w = -1.f;
        return coord;
    }  
    
    float D1 = determinantTetrahedron(mat, p, v[1], v[2], v[3]);
    float D2 = determinantTetrahedron(mat, v[0], p, v[2], v[3]);
    float D3 = determinantTetrahedron(mat, v[0], v[1], p, v[3]);
    float D4 = determinantTetrahedron(mat, v[0], v[1], v[2], p);
    
    coord.x = D1/D0;
    coord.y = D2/D0;
    coord.z = D3/D0;
    coord.w = D4/D0;
    
    return coord;
}

char pointInsideTetrahedronTest(const Vector3F & p, const Vector3F * v)
{
    BarycentricCoordinate coord = getBarycentricCoordinate(p, v);
    // std::cout<<"sum "<<coord.x + coord.y + coord.z + coord.w<<"\n";
    
    //Vector3F proof = v[0] * coord.x + v[1] * coord.y + v[2] * coord.z + v[3] * coord.w;
    //std::cout<<"proof "<<proof.str()<<"\n";
    
    if(coord.x < 0.f || coord.y < 0.f || coord.z < 0.f || coord.w < 0.f)
        return 0;
    
    if(coord.x > 1.f || coord.y > 1.f || coord.z > 1.f || coord.w > 1.f)
        return 0;
    
    return 1;
}

struct Simplex {
    Simplex() {
        d = 0;
        
    }
    int d;
    Vector3F p[4];
};

void addToSimplex(Simplex & s, const Vector3F & p)
{
    if(s.d < 1) {
        s.p[0] = p;
        s.d = 1;
    }
    else if(s.d < 2) {
        s.p[1] = p;
        s.d = 2;
    }
    else if(s.d < 3) {
        s.p[2] = p;
        s.d = 3;
    }
    else {
        s.p[3] = p;
        s.d = 4;
    }
}

void removeFromSimplex(Simplex & s, int ind)
{
    // std::cout<<"remove vertex "<<ind;   
    for(int i = ind; i < 3; i++) {
        s.p[i] = s.p[i+1];
    }
    s.d--;
}

Vector3F closestToOriginOnLine(const Vector3F & p0, const Vector3F & p1)
{
    // std::cout<<" closest on line "<<p0.str()<<" "<<p1.str()<<"\n";
    Vector3F p = p0;
    float lp = p.length();
    if(lp < 0.001f) return p;
    
    Vector3F dir = p1 - p0;
    //std::cout<<" dir "<<dir.str();
    if(dir.length() < 0.001f) return p;
    
    if(dir.dot(p) > 0.f) return p;
    
    p.normalize(); p.reverse();
    dir.normalize();
    
    const float factor = dir.dot(p);
    
    p = p0 + dir * p0.length() * factor;
    // std::cout<<" p "<<p.str()<<"\n";
    
    return p;
}

char closestPointToOriginInsideTriangle(Vector3F & onplane, const Vector3F & p0, const Vector3F & p1, const Vector3F & p2)
{
    Vector3F ab = p1 - p0;
    Vector3F ac = p2 - p0;
    Vector3F nor = ab.cross(ac);
    nor.normalize();
    
    float t = p0.dot(nor);
    onplane = nor * t;
    
    Vector3F e01 = p1 - p0;
	Vector3F x0 = onplane - p0;
	if(e01.cross(x0).dot(nor) < 0.f) return 0;
	
	Vector3F e12 = p2 - p1;
	Vector3F x1 = onplane - p1;
	if(e12.cross(x1).dot(nor) < 0.f) return 0;
	
	Vector3F e20 = p0 - p2;
	Vector3F x2 = onplane - p2;
	if(e20.cross(x2).dot(nor) < 0.f) return 0;
	
	return 1;
}

Vector3F closestToOriginOnTriangle(const Vector3F & a, const Vector3F & b, const Vector3F & c)
{
    // std::cout<<" closest on triangle "<<a.str()<<" "<<b.str()<<" "<<c.str()<<"\n";
    Vector3F onplane;
    if(closestPointToOriginInsideTriangle(onplane, a, b, c))
        return onplane;
    
    Vector3F online = closestToOriginOnLine(a, b);
    float minDistance = online.length();
    Vector3F l = closestToOriginOnLine(b, c);
    float ll = l.length();
    if(ll < minDistance) {
        online = l;
        minDistance = ll;
    }
    l = closestToOriginOnLine(c, a);
    ll = l.length();
    if(ll < minDistance) {
        online = l;
        minDistance = ll;
    }
    return online;
}

Vector3F closestToOriginOnTetrahedron(Simplex & s)
{
    Vector3F q = closestToOriginOnTriangle(s.p[0], s.p[1], s.p[2]);
    float d = q.length();
    float minD = d;
    Vector3F r = q;
    int farthest = 3;
    
    q = closestToOriginOnTriangle(s.p[0], s.p[1], s.p[3]);
    d = q.length();
    if(d < minD) {
        minD = d;
        r = q;
        farthest = 2;
    }
    
    q = closestToOriginOnTriangle(s.p[0], s.p[2], s.p[3]);
    d = q.length();
    if(d < minD) {
        minD = d;
        r = q;
        farthest = 1;
    }
    
    q = closestToOriginOnTriangle(s.p[1], s.p[2], s.p[3]);
    d = q.length();
    if(d < minD) {
        minD = d;
        r = q;
        farthest = 0;
    }
    
    removeFromSimplex(s, farthest); 
    
    return r;
}

Vector3F closestToOriginWithinSimplex(Simplex & s)
{
    if(s.d < 2) {
        return s.p[0];
    }
    if(s.d < 3) {
        return closestToOriginOnLine(s.p[0], s.p[1]);
    }
    if(s.d < 4) {
        return closestToOriginOnTriangle(s.p[0], s.p[1], s.p[2]);
    }
    
    return closestToOriginOnTetrahedron(s);
}

char isOriginInsideSimplex(const Simplex & s)
{
    if(s.d < 4) return 0;
    return pointInsideTetrahedronTest(Vector3F::Zero, s.p);
}

class PointSet {
public:
    Vector3F X[3];
    
    const Vector3F supportPoint(const Vector3F & v) const
    {
        Vector3F res = X[0];
        float mdotv = X[0].dot(v);
        float dotv = X[1].dot(v);
        if(dotv > mdotv) {
            mdotv = dotv;
            res = X[1];
        }
        dotv = X[2].dot(v);
        if(dotv > mdotv) {
            mdotv = dotv;
            res = X[2];
        }
        return res;
    }
};

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