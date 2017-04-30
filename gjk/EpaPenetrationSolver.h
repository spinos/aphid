#ifndef EPAPENETRATIONSOLVER_H
#define EPAPENETRATIONSOLVER_H

#include "Gjk.h"

class EpaPenetrationSolver {
public:
    EpaPenetrationSolver();
    
    void depth(const PointSet & A, const PointSet & B, ClosestTestContext * result);
};
#endif        //  #ifndef EPAPENETRATIONSOLVER_H

