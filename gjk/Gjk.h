#ifndef GJK_H
#define GJK_H

#include <AllMath.h>

#define TINY_VALUE 1e-6
#define MARGIN_DISTANCE 1e-4

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
	Vector3F pB[4];
    int d;
};

struct ClosestTestContext {
	Simplex W;
	BarycentricCoordinate contributes;
    Vector3F referencePoint;
    Vector3F rayDirection;
    Vector3F contactNormal;
	Vector3F contactPoint;
	float distance;
	char needContributes;
	char hasResult;
};

void closestOnLine(const Vector3F * p, ClosestTestContext * io);
void closestOnTriangle(const Vector3F * p, ClosestTestContext * io);
void closestPointToOriginInsideTriangle(const Vector3F * p, ClosestTestContext * io);
void closestOnTetrahedron(const Vector3F * p, ClosestTestContext * io);
void closestOnSimplex(Simplex & s, ClosestTestContext * io);

void resetSimplex(Simplex & s);
void addToSimplex(Simplex & s, const Vector3F & p);
void addToSimplex(Simplex & s, const Vector3F & p, const Vector3F & pb);
void removeFromSimplex(Simplex & s, BarycentricCoordinate coord);
char isOriginInsideSimplex(const Simplex & s);
char isPointInsideSimplex(const Simplex & s, const Vector3F & p);
Vector3F closestToOriginWithinSimplex(Simplex & s);
char pointInsideTetrahedronTest(const Vector3F & p, const Vector3F * v);
Vector3F supportMapping(const PointSet & A, const PointSet & B, const Vector3F & v);
Vector3F interpolatePointB(const Simplex & s, const BarycentricCoordinate & coord);
#endif        //  #ifndef GJK_H

