#ifndef TETRAHEDRON_MATH_H
#define TETRAHEDRON_MATH_H

#include <AllMath.h>
// http://pelopas.uop.gr/~nplatis%20/files/PlatisTheoharisRayTetra.pdf

struct PluckerCoordinate {
    Vector3F U, V;
    
    void set(const Vector3F & L, const Vector3F & P) {
        U = L;
        V = L.cross(P);
    }
    
    float dot(const PluckerCoordinate & s) const {
        return U.dot(s.V) + s.U.dot(V);
    }
};

inline int getSign(float d) {
    if(d> 0.f) return 1;
    if(d< 0.f) return -1;
    return 0;
}

inline bool tetrahedronLineIntersection(Vector3F * tet, const Vector3F & lineBegin, const Vector3F & lineEnd)
{
    PluckerCoordinate pieR, pieIj;
    pieR.set(lineEnd - lineBegin, lineBegin);
    
    int faceEnter = -1;
    int faceLeave = -1;
    
    int thetaI[3];
    float w[3];
    int i;
    for(i=3; i>=0; i--) {
        pieIj.set(tet[i]-tet[0], tet[i]);
        w[0] = pieR.dot(pieIj);
        thetaI[0] = getSign(w[0]);
        
        pieIj.set(tet[i]-tet[1], tet[i]);
        w[1] = pieR.dot(pieIj);
        thetaI[1] = getSign(w[1]);
        
        pieIj.set(tet[i]-tet[2], tet[i]);
        w[2] = pieR.dot(pieIj);
        thetaI[2] = getSign(w[2]);
        
        if(thetaI[0] !=0 || thetaI[1] !=0 || thetaI[2] !=0) {
            if(faceEnter < 0 && thetaI[0]>=0 && thetaI[1]>=0 && thetaI[2]>=0)
                faceEnter = i;
            else if(faceLeave < 0 && thetaI[0]<=0 && thetaI[1]<=0 && thetaI[2]<=0)
                faceLeave = i;
        }
        
        if(faceEnter >=0 && faceLeave >= 0) {
            std::cout<<" sum "<<w[0]+w[1]+w[2]<<" ";
            return true;
        }
    }
    
    return false;
}
#endif        //  #ifndef TETRAHEDRON_MATH_H

