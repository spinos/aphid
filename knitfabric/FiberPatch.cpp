#include "FiberPatch.h"
#include <BezierCurve.h>

FiberPatch::FiberPatch() 
{
    m_indices = 0;
    m_fiberP = 0;
	m_halfThickness = 0.5f;
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
    m_numPointPerFiber = numPointPerYarn * 2;
    m_fiberP = new Vector3F[m_numFiber * m_numPointPerFiber];
    m_indices = new unsigned[m_numPointPerFiber];
    for(unsigned i = 0; i < m_numPointPerFiber; i++) m_indices[i] = i;
}

void FiberPatch::processYarn(unsigned iyarn, Vector3F * yarnP, Vector3F * yarnN, Vector3F * yarnT, unsigned nppy)
{
	unsigned i;
	BezierCurve cuv;
	for(i = 0; i < nppy; i++) cuv.addVertex(yarnP[i]);
	cuv.computeKnots();

	const float delta = 1.f / (float)m_numPointPerFiber;

    for(unsigned ifiber = 0; ifiber < numFiberPerYarn(); ifiber++) {
        Vector3F * p = fiberAt(iyarn, ifiber);
        for(i = 0; i < m_numPointPerFiber; i++) {
			Vector3F q = cuv.interpolate(delta * i, cuv.m_cvs);
			Vector3F nor = cuv.interpolate(delta * i, yarnN);
			Vector3F tang = cuv.interpolate(delta * i, yarnT);
				
            if(ifiber < 3) {
				nor.rotateAroundAxis(tang, 2.f * ifiber + .55f * i);
				p[i] = q + nor * m_halfThickness * 0.5f;
			}
			else {
				nor.rotateAroundAxis(tang, 0.69f * ifiber + .55f * i);
				p[i] = q + nor * m_halfThickness;
			}
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

void FiberPatch::setThickness(float thickness)
{
	m_halfThickness = thickness * .5f;
}
