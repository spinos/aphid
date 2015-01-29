#include "EpaPenetrationSolver.h"

EpaPenetrationSolver::EpaPenetrationSolver() {}
    
void EpaPenetrationSolver::depth(const PointSet & A, const PointSet & B, ClosestTestContext * result)
{
    Vector3F w, pa, pb;
    Vector3F localA, localB;
    
    result->referencePoint = Vector3F::Zero;
    //closestOnSimplex(result);
    //smallestSimplex(result);
    Vector3F v = result->contactNormal;
    //for(int i=0; i < 99; i++) {
        pa = A.supportPoint(v, result->transformA, localA);
        pb = B.supportPoint(v.reversed(), result->transformB, localB);
        w = pa - pb;
        
        
        result->referencePoint = w;
        
        closestOnSimplex(result);
        v = w - result->closestPoint;
        //if(v.length2() < 0.001f) break;
        
        //smallestSimplex(result);
        
        addToSimplex(result->W, w, localB);
    //}
    result->contactNormal = w;
	result->hasResult = 1;
}
