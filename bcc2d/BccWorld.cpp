#include "BccWorld.h"
#include <BezierCurve.h>
#include <LineDrawer.h> 
BccWorld::BccWorld(LineDrawer * drawer) 
{
    m_drawer = drawer;
    m_curve = new BezierCurve;
    m_curve->createVertices(9);
    m_curve->m_cvs[0].set(2.f + RandomFn11(), 2.f + RandomFn11(), 0.f);
    m_curve->m_cvs[1].set(2.f + RandomFn11(), 9.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[2].set(10.f + RandomFn11(), 9.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[3].set(11.f + RandomFn11(), 2.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[4].set(19.f + RandomFn11(), 2.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[5].set(21.f + RandomFn11(), 6.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[6].set(18.f + RandomFn11(), 12.2f + RandomFn11(), 0.f);
    m_curve->m_cvs[7].set(12.f + RandomFn11(), 12.2f + RandomFn11(), 0.f);
    m_curve->m_cvs[8].set(13.f + RandomFn11(), 8.2f + RandomFn11(), 0.f);
    m_curve->computeKnots(); 
    
    const unsigned ns = m_curve->numSegments();
    m_segmentCurve = new BezierCurve[ns];
    
    for(unsigned i=0; i < ns; i++) m_segmentCurve[i].createVertices(4);
    m_curve->getAccSegmentCurves(m_segmentCurve);
    for(unsigned i=0; i < ns; i++) m_segmentCurve[i].computeKnots();
}

BccWorld::~BccWorld() {}
 
void BccWorld::draw()
{
    glColor3f(.1f, .1f, .1f); 
    m_drawer->linearCurve(*m_curve);
    glColor3f(.98f, .2f, .1f);
    m_drawer->smoothCurve(*m_curve, 32);
    
    BoundingBox box; 
    m_curve->getAabb(&box);
    
    // glColor3f(.21f, .21f, .21f);
    // m_drawer->boundingBox(box); 
    
    const unsigned ns = m_curve->numSegments();
    unsigned i;
    for(i=0; i < ns; i++) {
        glColor3f(.1f, .1f, .1f); 
        m_drawer->linearCurve(m_segmentCurve[i]);
        glColor3f(.1f, .99f, .1f);
        m_drawer->smoothCurve(m_segmentCurve[i], 8);
        
        box.reset();
        m_segmentCurve[i].getAabb(&box);
        
        glColor3f(.21f, .21f, .21f);
        m_drawer->boundingBox(box); 
    }
} 
