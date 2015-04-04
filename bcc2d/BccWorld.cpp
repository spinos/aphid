#include "BccWorld.h"
#include <BezierCurve.h>
#include <GeoDrawer.h>
#include "BccGrid.h"

BccWorld::BccWorld(GeoDrawer * drawer) 
{
    m_testP.set(10.f, 12.f, 0.f);
    m_drawer = drawer;  
    m_curve = new BezierCurve;
    m_curve->createVertices(9);
    m_curve->m_cvs[0].set(4.f + RandomFn11(), 1.f + RandomFn11(), 0.f);
    m_curve->m_cvs[1].set(2.f + RandomFn11(), 9.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[2].set(12.f + RandomFn11(), 12.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[3].set(9.f + RandomFn11(), 2.4f + RandomFn11(), 0.f);
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
    
    BoundingBox box; 
    m_curve->getAabb(&box);
    
    m_grid = new BccGrid(box);
    m_grid->create(m_curve);
}

BccWorld::~BccWorld() {}
 
void BccWorld::draw()
{
    glColor3f(.1f, .1f, .1f); 
    // m_drawer->linearCurve(*m_curve);
    glColor3f(.0f, .3f, .5f);
    m_drawer->smoothCurve(*m_curve, 32);
    
    BoundingBox box;
    m_grid->getBounding(box);
    
    glColor3f(.21f, .21f, .21f);
    m_drawer->boundingBox(box);
    
    m_grid->draw(m_drawer);
}

void BccWorld::testSegments()
{
    BoundingBox box; 
    const unsigned ns = m_curve->numSegments();
    unsigned i;
    for(i=0; i < ns; i++) {
        glColor3f(.1f, .1f, .1f); 
        // m_drawer->linearCurve(m_segmentCurve[i]);
        glColor3f(.1f, .99f, .1f);
        m_drawer->smoothCurve(m_segmentCurve[i], 8);
        
        box.reset();
        m_segmentCurve[i].getAabb(&box);
        
        glColor3f(.21f, .21f, .21f);
        m_drawer->boundingBox(box); 
    }
}

void BccWorld::testDistanctToCurve()
{
    Vector3F cls;
    float minD = 1e8;
    const unsigned ns = m_curve->numSegments();
    unsigned i;
    for(i=0; i < ns; i++) {
        SimpleBezierSpline sp;
        sp.cv[0] = m_segmentCurve[i].m_cvs[0];
        sp.cv[1] = m_segmentCurve[i].m_cvs[1];
        sp.cv[2] = m_segmentCurve[i].m_cvs[2];
        sp.cv[3] = m_segmentCurve[i].m_cvs[3];
        testDistanceToPoint(sp, m_testP, minD, cls);
    }
    
    m_drawer->setColor(.1f, 1.f, 0.f);
    m_drawer->arrow(m_testP, cls);
    
    m_curve->distanceToPoint(m_testP, cls);
    m_drawer->arrow(m_testP, cls);
}

void BccWorld::testDistanceToPoint(SimpleBezierSpline & spline, const Vector3F & pnt, float & minDistance, Vector3F & closestP)
{
    float paramMin = 0.f;
    float paramMax = 1.f;
    Vector3F line[2];
    
    line[0] = spline.calculateBezierPoint(paramMin);
    line[1] = spline.calculateBezierPoint(paramMax);
    
    Vector3F pol;
    float t;
    for(;;) {
        glColor3f(1.f, .9f, .0f);
        glBegin(GL_LINES);
        glVertex3fv((GLfloat *)&line[0]);
        glVertex3fv((GLfloat *)&line[1]);
        glEnd();
        
        float d = closestDistanceToLine(line, pnt, pol, t);
        
        const float tt = paramMin * (1.f - t) + paramMax * t;
        
// end of line
        if((tt <= 0.f || tt >= 1.f) && paramMax - paramMin < 0.1f) {
            if(d < minDistance) {
                minDistance = d;
                closestP = pol;
            }
            break;
        }
        
        const float h = .5f * (paramMax - paramMin);

// small enought        
        if(h < 1e-3) {
            if(d < minDistance) {
                minDistance = d;
                closestP = pol;
            }
            break;
        }
        
        if(t > .5f)
            paramMin = tt - h * .5f;
        else
            paramMax = tt + h * .5f;
            
        line[0] = spline.calculateBezierPoint(paramMin);
        line[1] = spline.calculateBezierPoint(paramMax);
    
    }
}

void BccWorld::moveTestP(float x, float y, float z)
{ m_testP += Vector3F(x, y, z);}

