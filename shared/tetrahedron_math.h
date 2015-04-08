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

inline bool tetrahedronLineIntersection(const Vector3F * tet, const Vector3F & lineBegin, const Vector3F & lineEnd,
    Vector3F & enterP)
{
    PluckerCoordinate pieR, pieIj;
    pieR.set(lineBegin, lineEnd);
    
    int faceEnter = -1;
    int faceLeave = -1;
    
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
            else if(faceLeave < 0 && thetaI[0]<=0 && thetaI[1]<=0 && thetaI[2]<=0)
                faceLeave = i;
        }
        
        if(faceEnter > -1) {
            sumW = w[0]+w[1]+w[2];
            enterP = tet[ TetrahedronToTriangleVertexByFace[i][2] ] * w[0] / sumW 
                + tet[ TetrahedronToTriangleVertexByFace[i][0] ] * w[1] / sumW
                + tet[ TetrahedronToTriangleVertexByFace[i][1] ] * w[2] / sumW;
                
            if(enterP.distanceTo(lineBegin) > lineEnd.distanceTo(lineBegin)) return false;
            
            return true;
        }
    }
// no enter point    
    return false;
}
#endif        //  #ifndef TETRAHEDRON_MATH_H

