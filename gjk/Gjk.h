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

struct Aabb {
	Vector3F low, high;
};

void resetAabb(Aabb & dst);
void expandAabb(Aabb & dst, const Vector3F & p);

class PointSet {
public:
	virtual ~PointSet() {}
    Vector3F X[3];
    
    virtual const float angularMotionDisc() const
    {
		Aabb box;
		resetAabb(box);
        for(int i=0; i < 3; i++) 
			expandAabb(box, X[i]);

        const Vector3F center = box.low * 0.5f + box.high * 0.5f;
        const Vector3F d = box.high - box.low;
        return center.length() + d.length() * 0.5f;
    }
    
    virtual const Vector3F supportPoint(const Vector3F & v, const Matrix44F & space, Vector3F & localP) const
    {
        float maxdotv = -1e8;
        float dotv;
        
        Vector3F res;
        Vector3F worldP;
        
        for(int i=0; i < 3; i++) {
            worldP = space.transform(X[i]);
            dotv = worldP.dot(v);
            if(dotv > maxdotv) {
                maxdotv = dotv;
                res = worldP;
                localP = X[i];
            }
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
	Matrix44F transformA, transformB;
	BarycentricCoordinate contributes;
	Vector3F referencePoint;
    Vector3F rayDirection;
    Vector3F contactPointB;
	Vector3F closestPoint;
	Vector3F separateAxis;
	float distance;
	char needContributes;
	char hasResult;
};

struct ContinuousCollisionContext {
    Quaternion orientationA;
	Quaternion orientationB;
	Vector3F positionA;
	Vector3F positionB;
    Vector3F linearVelocityA;
	Vector3F linearVelocityB;
	Vector3F angularVelocityA;
	Vector3F angularVelocityB;
	Vector3F contactNormal;
	Vector3F contactPointB;
	float TOI;
	float penetrateDepth;
	char hasContact;
};

void closestOnSimplex(ClosestTestContext * io);
void smallestSimplex(ClosestTestContext * io);
void interpolatePointB(ClosestTestContext * io);

void resetSimplex(Simplex & s);
void addToSimplex(Simplex & s, const Vector3F & p);
void addToSimplex(Simplex & s, const Vector3F & p, const Vector3F & pb);
char isPointInsideSimplex(const Simplex & s, const Vector3F & p);
#endif        //  #ifndef GJK_H

