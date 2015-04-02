#include "BccWorld.h"
#include <BezierCurve.h>
#include <LineDrawer.h>
BccWorld::BccWorld(LineDrawer * drawer) 
{
    m_drawer = drawer;
    m_curve = new BezierCurve;
    m_curve->createVertices(7);
    m_curve->m_cvs[0].set(2.f + RandomFn11(), 2.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[1].set(2.f + RandomFn11(), 3.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[2].set(3.f + RandomFn11(), 3.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[3].set(4.f + RandomFn11(), 2.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[4].set(5.f + RandomFn11(), 1.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[5].set(3.f + RandomFn11(), 1.2f + RandomFn11(), 0.f);
    m_curve->m_cvs[6].set(2.3f + RandomFn11(), 2.2f + RandomFn11(), 0.f);
    m_curve->computeKnots();
    
    const unsigned ns = m_curve->numSegments();
    m_segmentCurve = new BezierCurve[ns];
    
    for(unsigned i=0; i < ns; i++) m_segmentCurve[i].createVertices(4);
    m_curve->getSegmentCurves(m_segmentCurve);
    for(unsigned i=0; i < ns; i++) m_segmentCurve[i].computeKnots();
}

BccWorld::~BccWorld() {}

void BccWorld::draw() 
{
    m_drawer->linearCurve(*m_curve);
    m_drawer->smoothCurve(*m_curve, 8);
    
    BoundingBox box;
    m_curve->getAabb(&box);
    
    m_drawer->boundingBox(box);
    
    const unsigned ns = m_curve->numSegments();
    unsigned i;
    for(i=0; i < 2; i++) {
        m_drawer->linearCurve(m_segmentCurve[i]);
        m_drawer->smoothCurve(m_segmentCurve[i], 8);
    }
}
