#ifndef GJK_H
#define GJK_H

#include <AllMath.h>

#define TINY_VALUE 1e-6

struct BarycentricCoordinate {
    float x, y, z, w;
};

float determinantTetrahedron(Matrix44F & mat, const Vector3F & v1, const Vector3F & v2, const Vector3F & v3, const Vector3F & v4);

BarycentricCoordinate getBarycentricCoordinate2(const Vector3F & p, const Vector3F * v);
BarycentricCoordinate getBarycentricCoordinate3(const Vector3F & p, const Vector3F * v);
BarycentricCoordinate getBarycentricCoordinate4(const Vector3F & p, const Vector3F * v);

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

struct Simplex {
    Vector3F p[4];
    int d;
};

Vector3F closestToOriginOnLine(const Vector3F & p0, const Vector3F & p1);

char closestPointToOriginInsideTriangle(Vector3F & onplane, const Vector3F & p0, const Vector3F & p1, const Vector3F & p2);

Vector3F closestToOriginOnTriangle(const Vector3F & a, const Vector3F & b, const Vector3F & c);

Vector3F closestToOriginOnTetrahedron(const Vector3F * p);
Vector3F closestToOriginOnTetrahedron(Simplex & s);

void resetSimplex(Simplex & s);
void addToSimplex(Simplex & s, const Vector3F & p);
void removeFromSimplex(Simplex & s, BarycentricCoordinate coord);
char isOriginInsideSimplex(const Simplex & s);
Vector3F closestToOriginOnLine(Simplex & s);
Vector3F closestToOriginOnTriangle(Simplex & s);
Vector3F closestToOriginWithinSimplex(Simplex & s);
char pointInsideTetrahedronTest(const Vector3F & p, const Vector3F * v);
#endif        //  #ifndef GJK_H

