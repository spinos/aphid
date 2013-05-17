#pragma once

#include <AllMath.h>

class FiberPatch {
public:
    FiberPatch();
    virtual ~FiberPatch();
    void cleanup();
    unsigned numFiberPerYarn() const;
    void create(unsigned numYarn, unsigned numPointPerYarn);
    
    void processYarn(unsigned iyarn, Vector3F * yarnP, Vector3F * yarnN, Vector3F * yarnT);
    
    Vector3F * fiberAt(unsigned iyarn, unsigned ifiber);
    Vector3F * fiberAt(unsigned ifiber);
    
    unsigned getNumFiber() const;
    unsigned numPointsPerFiber() const;
    unsigned * fiberIndices();
private:
    unsigned * m_indices;
    Vector3F * m_fiberP;
    unsigned m_numFiber;
    unsigned m_numPointPerFiber;
};

