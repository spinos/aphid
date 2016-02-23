#ifndef TETRAHEDRON_MATH_H
#define TETRAHEDRON_MATH_H

#include <AllMath.h>

/* 
*   3
*  /|\
* / | \
*2-----1
* \ | /
*  \|/
*   0 
*/

namespace aphid {

static const int TetrahedronToTriangleVertex[12] = {
0, 1, 2, 
1, 0, 3,
2, 3, 0,
3, 2, 1};

static const int TetrahedronToTriangleVertexByFace[4][3] = {
{0, 1, 2},
{1, 0, 3},
{2, 3, 0},
{3, 2, 1}};

// http://pelopas.uop.gr/~nplatis%20/files/PlatisTheoharisRayTetra.pdf

struct PluckerCoordinate {
    void set(const Vector3F & P0, const Vector3F & P1) {
        U = P1 - P0;
        V = U.cross(P0);
    }
    
    float dot(const PluckerCoordinate & s) const {
        return U.dot(s.V) + s.U.dot(V);
    }
    
    Vector3F U, V;
};

inline float determinantTetrahedron(Matrix44F & mat, const Vector3F & v1, const Vector3F & v2, const Vector3F & v3, const Vector3F & v4)
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

inline Float4 getBarycentricCoordinate4(const Vector3F & p, const Vector3F * v)
{
    Matrix44F mat;
    
    Float4 coord;
    
    float D0 = determinantTetrahedron(mat, v[0], v[1], v[2], v[3]);
    if(D0 == 0.f) {
        std::cout<<" tetrahedron is degenerate ("<<v[0].str()<<","<<v[1].str()<<","<<v[2].str()<<","<<v[3].str()<<")\n";
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

inline bool pointInsideTetrahedronTest(const Vector3F & p, const Vector3F * v)
{        
    Float4 coord = getBarycentricCoordinate4(p, v);
   
    if(coord.x < 0.f || coord.y < 0.f || coord.z < 0.f || coord.w < 0.f)
        return false;
    
    if(coord.x > 1.f || coord.y > 1.f || coord.z > 1.f || coord.w > 1.f)
        return false;
    
    return true;
}

inline bool tetrahedronLineIntersection(const Vector3F * tet, const Vector3F & lineBegin, const Vector3F & lineEnd,
    Vector3F & enterP)
{
    if(pointInsideTetrahedronTest(lineBegin, tet)) return true;
    if(pointInsideTetrahedronTest(lineEnd, tet)) return true;
    
    PluckerCoordinate pieR, pieIj;
    pieR.set(lineBegin, lineEnd);
    
    int faceEnter = -1;
    // int faceLeave = -1;
    
    int thetaI[3];
    float w[3], sumW;
    int i;
    for(i=3; i>=0; i--) {
        pieIj.set(tet[ TetrahedronToTriangleVertexByFace[i][0] ], tet[ TetrahedronToTriangleVertexByFace[i][1] ]);
        w[0] = pieR.dot(pieIj);
        thetaI[0] = GetSign(w[0]);
        
        pieIj.set(tet[ TetrahedronToTriangleVertexByFace[i][1] ], tet[ TetrahedronToTriangleVertexByFace[i][2] ]);
        w[1] = pieR.dot(pieIj);
        thetaI[1] = GetSign(w[1]);
        
        pieIj.set(tet[ TetrahedronToTriangleVertexByFace[i][2] ], tet[ TetrahedronToTriangleVertexByFace[i][0] ]);
        w[2] = pieR.dot(pieIj);
        thetaI[2] = GetSign(w[2]);
        
        if(thetaI[0] !=0 || thetaI[1] !=0 || thetaI[2] !=0) {
            if(faceEnter < 0 && thetaI[0]>=0 && thetaI[1]>=0 && thetaI[2]>=0)
                faceEnter = i;
            //else if(faceLeave < 0 && thetaI[0]<=0 && thetaI[1]<=0 && thetaI[2]<=0)
              //  faceLeave = i;
        }
        
        // if(faceLeave > -1) {
             // sumW = w[0]+w[1]+w[2];
             // enterP = tet[ TetrahedronToTriangleVertexByFace[i][2] ] * w[0] / sumW 
                // + tet[ TetrahedronToTriangleVertexByFace[i][0] ] * w[1] / sumW
                // + tet[ TetrahedronToTriangleVertexByFace[i][1] ] * w[2] / sumW;
             // if((enterP-lineBegin).dot(lineEnd - lineBegin) < 0.f) return false;
             // if(enterP.distanceTo(lineBegin) < lineEnd.distanceTo(lineBegin)) return true; 
        // }
        
        if(faceEnter > -1) {
            sumW = w[0]+w[1]+w[2];
            enterP = tet[ TetrahedronToTriangleVertexByFace[i][2] ] * w[0] / sumW 
                + tet[ TetrahedronToTriangleVertexByFace[i][0] ] * w[1] / sumW
                + tet[ TetrahedronToTriangleVertexByFace[i][1] ] * w[2] / sumW;
            if((enterP-lineBegin).dot(lineEnd - lineBegin) < 0.f) return false;   
            if(enterP.distanceTo(lineBegin) > lineEnd.distanceTo(lineBegin)) return false;
            
            return true;
        }
    }
// no enter point    
    return false;
}

inline bool isTetrahedronDegenerated(const Vector3F * p)
{
    Matrix44F mat;
    float D0 = determinantTetrahedron(mat, p[0], p[1], p[2], p[3]);
	if(D0 < 0.f) D0 = -D0;
    return (D0 < 1e-5);
}

inline float tetrahedronVolume(const Vector3F * p) 
{
    Vector3F e1 = p[1]-p[0];
	Vector3F e2 = p[2]-p[0];
	Vector3F e3 = p[3]-p[0];
	return  e1.dot( e2.cross( e3 ) ) / 6.0f;
}

inline bool isTriangleDegenerated(const Vector3F * v)
{
	const Vector3F n = Vector3F(v[1], v[0]).cross( Vector3F(v[2], v[0]) );
    const float D0 = n.length2();
	// std::cout<<" D0 "<<D0<<"\n";
    return (D0 < 1e-6f);
}

inline Vector3F closestPOnLine(const Vector3F * p, const Vector3F & referencePoint)
{
    Vector3F vr = referencePoint - p[0];
    Vector3F v1 = p[1] - p[0];
	const float dr = vr.length();
	if(dr < 1e-6f) {
        return p[0];
    }
	
	const float d1 = v1.length();
	vr.normalize();
	v1.normalize();
	float vrdv1 = vr.dot(v1) * dr;
	if(vrdv1 < 0.f) vrdv1 = 0.f;
	if(vrdv1 > d1) vrdv1 = d1;
	
	v1 = p[0] + v1 * vrdv1;
    return v1;
}

inline bool closestPointToOriginInsideTriangle(const Vector3F * p, const Vector3F & referencePoint,
                                                   Vector3F & resultP)
{
	Vector3F ab = p[1] - p[0];
    Vector3F ac = p[2] - p[0];
    Vector3F nor = ab.cross(ac);
    nor.normalize();
    
    float t = p[0].dot(nor);
    Vector3F onplane = nor * t;
    
    Vector3F e01 = p[1] - p[0];
	Vector3F x0 = onplane - p[0];
	if(e01.cross(x0).dot(nor) < 0.f) return false;
	
	Vector3F e12 = p[2] - p[1];
	Vector3F x1 = onplane - p[1];
	if(e12.cross(x1).dot(nor) < 0.f) return false;
	
	Vector3F e20 = p[0] - p[2];
	Vector3F x2 = onplane - p[2];
	if(e20.cross(x2).dot(nor) < 0.f) return false;
	
	resultP = onplane + referencePoint;
    return true;
}

inline Vector3F closestPOnTriangle(const Vector3F * p, const Vector3F & referencePoint)
{	
	Vector3F pr[3];
	pr[0] = p[0] - referencePoint;
	pr[1] = p[1] - referencePoint;
	pr[2] = p[2] - referencePoint;
	
    Vector3F result;
    if(closestPointToOriginInsideTriangle(pr, referencePoint, result))
        return result;
		
    pr[0] = p[0];
	pr[1] = p[1];
	result = closestPOnLine(pr, referencePoint);
    float minDist = result.distanceTo(referencePoint);
	
	pr[0] = p[1];
	pr[1] = p[2];
	Vector3F q = closestPOnLine(pr, referencePoint);
    float dist = q.distanceTo(referencePoint);
    if(minDist > dist) {
        result = q;
        minDist = dist;
    }
	
	pr[0] = p[2];
	pr[1] = p[0];
	q = closestPOnLine(pr, referencePoint);
    dist = q.distanceTo(referencePoint);
    if(minDist > dist) {
        result = q;
        minDist = dist;
    }
    return result;
}

inline Vector3F closestPOnTetrahedron(const Vector3F * p, const Vector3F & referencePoint)
{
	Vector3F result = closestPOnTriangle(p, referencePoint);
    float minDist = result.distanceTo(referencePoint);
	
	Vector3F pr[3];
	pr[0] = p[0];
	pr[1] = p[1];
	pr[2] = p[3];
	Vector3F q = closestPOnTriangle(pr, referencePoint);
    float dist = q.distanceTo(referencePoint);
	if(minDist > dist) {
        result = q;
        minDist = dist;
    }
    
	pr[0] = p[0];
	pr[1] = p[2];
	pr[2] = p[3];
	q = closestPOnTriangle(pr, referencePoint);
	dist = q.distanceTo(referencePoint);
	if(minDist > dist) {
        result = q;
        minDist = dist;
    }
    
	pr[0] = p[1];
	pr[1] = p[2];
	pr[2] = p[3];
	q = closestPOnTriangle(pr, referencePoint);
    dist = q.distanceTo(referencePoint);
	if(minDist > dist) {
        result = q;
        minDist = dist;
    }
    
    return result;
}

inline Float4 getBarycentricCoordinate2(const Vector3F & p, const Vector3F * v)
{
	Float4 coord;
	Vector3F dv = v[1] - v[0];
	float D0 = dv.length();
	if(D0 < 1e-6f) {
        std::cout<<" line is degenerate ("<<v[0].str()<<","<<v[1].str()<<")\n";
        coord.x = coord.y = coord.z = coord.w = -1.f;
        return coord;
    }  
	Vector3F dp = p - v[1];
	coord.x = dp.length() / D0;
	coord.y = 1.f - coord.x;
	coord.z = coord.w = 0.f;
	return coord;
}

inline Float4 getBarycentricCoordinate3(const Vector3F & p, const Vector3F * v)
{
    Matrix33F mat;
    
    Float4 coord;
    
    Vector3F n = Vector3F(v[1] - v[0]).cross( Vector3F(v[2] - v[0]) );
	
    float D0 = n.length2();
    if(D0 < 1e-10f) {
        std::cout<<" tiangle is degenerate ("<<v[0].str()<<","<<v[1].str()<<","<<v[2].str()<<")\n";
        coord.x = coord.y = coord.z = coord.w = -1.f;
        return coord;
    }  
    
	Vector3F na = Vector3F(v[2] - v[1]).cross( Vector3F(p - v[1]) );
    float D1 = n.dot(na);
	Vector3F nb = Vector3F(v[0] - v[2]).cross( Vector3F(p - v[2]) );  
    float D2 = n.dot(nb);
	Vector3F nc = Vector3F(v[1] - v[0]).cross( Vector3F(p - v[0]) );
    float D3 = n.dot(nc);
    
    coord.x = D1/D0;
    coord.y = D2/D0;
    coord.z = D3/D0;
    coord.w = 0.f;
    
    return coord;
}

}
#endif        //  #ifndef TETRAHEDRON_MATH_H

