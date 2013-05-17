#include "FiberPatch.h"

FiberPatch::FiberPatch() 
{
    m_indices = 0;
    m_fiberP = 0;
}

FiberPatch::~FiberPatch() 
{
   cleanup();
}

void FiberPatch::cleanup()
{
    if(m_indices) delete[] m_indices;
    if(m_fiberP) delete[] m_fiberP;
}

unsigned FiberPatch::numFiberPerYarn() const
{
    return 12;
}

void FiberPatch::create(unsigned numYarn, unsigned numPointPerYarn)
{
    cleanup();
    
    m_numFiber = numYarn * numFiberPerYarn();
    m_numPointPerFiber = numPointPerYarn;
    m_fiberP = new Vector3F[m_numFiber * m_numPointPerFiber];
    m_indices = new unsigned[m_numPointPerFiber];
    for(unsigned i = 0; i < m_numPointPerFiber; i++) m_indices[i] = i;
}

void FiberPatch::processYarn(unsigned iyarn, Vector3F * yarnP, Vector3F * yarnN, Vector3F * yarnT)
{
    for(unsigned ifiber = 0; ifiber < numFiberPerYarn(); ifiber++) {
        Vector3F * p = fiberAt(iyarn, ifiber);
        for(unsigned i = 0; i < m_numPointPerFiber; i++) {
            Vector3F d = yarnN[i];
            d.rotateAroundAxis(yarnT[i], 0.5f * ifiber + 0.9f * i);
            p[i] = yarnP[i] + d * 0.09f;
        }
    }
}

Vector3F * FiberPatch::fiberAt(unsigned iyarn, unsigned ifiber)
{
    Vector3F * p = m_fiberP;
    p += iyarn * numFiberPerYarn() * m_numPointPerFiber + ifiber * m_numPointPerFiber;
    return p;
}

Vector3F * FiberPatch::fiberAt(unsigned ifiber)
{
    Vector3F * p = m_fiberP;
    p += ifiber * m_numPointPerFiber;
    return p;
}

unsigned FiberPatch::getNumFiber() const
{
    return m_numFiber;
}

unsigned FiberPatch::numPointsPerFiber() const
{
    return m_numPointPerFiber;
}

unsigned * FiberPatch::fiberIndices()
{
    return m_indices;
}
