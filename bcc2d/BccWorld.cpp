#include "BccWorld.h"
#include <BezierCurve.h>
#include <GeoDrawer.h>
#include "BccGrid.h"
#include <BaseBuffer.h>

BccWorld::BccWorld(GeoDrawer * drawer) 
{
    m_testP.set(10.f, 12.f, 0.f);
    m_drawer = drawer;  
    m_curve = new BezierCurve;
    m_curve->createVertices(9);
    m_curve->m_cvs[0].set(4.f + RandomFn11(), 1.f + RandomFn11(), 0.f);
    m_curve->m_cvs[1].set(2.f + RandomFn11(), 9.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[2].set(12.f + RandomFn11(), 8.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[3].set(6.f + RandomFn11(), .4f + RandomFn11(), 0.f);
    m_curve->m_cvs[4].set(19.f + RandomFn11(), 2.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[5].set(21.f + RandomFn11(), 6.4f + RandomFn11(), 0.f);
    m_curve->m_cvs[6].set(18.f + RandomFn11(), 12.2f + RandomFn11(), 0.f);
    m_curve->m_cvs[7].set(12.f + RandomFn11(), 12.2f + RandomFn11(), 0.f);
    m_curve->m_cvs[8].set(13.f + RandomFn11(), 8.2f + RandomFn11(), 0.f);
    m_curve->computeKnots(); 
    
    const unsigned ns = m_curve->numSegments();
    
    BoundingBox box; 
    m_curve->getAabb(&box);
    
    m_grid = new BccGrid(box);
    m_grid->create(m_curve);
	
	m_splines = new BaseBuffer;
	m_splines->create(ns * sizeof(BezierSpline));
	BezierSpline * spline = (BezierSpline *)m_splines->data();
	for(unsigned i=0; i < ns; i++)
		m_curve->getSegmentSpline(i, spline[i]);
}

BccWorld::~BccWorld() {}
 
void BccWorld::draw()
{
    glColor3f(.99f, .03f, .05f);
    m_drawer->smoothCurve(*m_curve, 32);
    
    BoundingBox box;
    m_grid->getBounding(box);
    
    glColor3f(.21f, .21f, .21f);
    m_drawer->boundingBox(box);
    
    m_grid->draw(m_drawer);
}

void BccWorld::testDistanctToCurve()
{
    Vector3F cls;
    float minD = 1e8;
	BezierSpline * spline = (BezierSpline *)m_splines->data();
    const unsigned ns = m_curve->numSegments();
    unsigned i;
    for(i=0; i < ns; i++) {
        testDistanceToPoint(spline[i], m_testP, minD, cls);
    }
    
    m_drawer->setColor(.1f, .9f, 0.f);
    m_drawer->arrow(m_testP, cls);
    
	m_drawer->setColor(.9f, .1f, 0.f);
    m_curve->distanceToPoint(m_testP, cls);
    m_drawer->arrow(m_testP, cls);
}

void BccWorld::testDistanceToPoint(BezierSpline & spline, const Vector3F & pnt, float & minDistance, Vector3F & closestP)
{
	BoundingBox box;
	box.expandBy(spline.cv[0]);
	box.expandBy(spline.cv[1]);
	box.expandBy(spline.cv[2]);
	box.expandBy(spline.cv[3]);
	
	if(box.distanceTo(pnt) > minDistance) return;
	
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

// small enough       
        if(h < 1e-3) {
            if(d < minDistance) {
                minDistance = d;
                closestP = pol;
            }
            break;
        }
        
        if(t > .5f)
            paramMin = tt - h * ((t - .5f)/.5f * .5f + .5f);
        else
            paramMax = tt + h * ((.5f - t)/.5f * .5f + .5f);
            
        line[0] = spline.calculateBezierPoint(paramMin);
        line[1] = spline.calculateBezierPoint(paramMax);
    
    }
}

void BccWorld::moveTestP(float x, float y, float z)
{ m_testP += Vector3F(x, y, z);}

void BccWorld::testSpline()
{
	glColor3f(.1f, .1f, .1f); 
    m_drawer->linearCurve(*m_curve);
    
	BezierSpline * spline = (BezierSpline *)m_splines->data();
	BezierSpline a, b;
	const unsigned ns = m_curve->numSegments();
	unsigned i, j;
	Vector3F p;

	for(i=0; i< ns; i++) {
		glColor3f(.1f, .1f, .1f);
		glBegin(GL_LINE_STRIP);
		glVertex3fv((GLfloat *)&spline[i].cv[0]);
		glVertex3fv((GLfloat *)&spline[i].cv[1]);
		glVertex3fv((GLfloat *)&spline[i].cv[2]);
		glVertex3fv((GLfloat *)&spline[i].cv[3]);
		glEnd();
		
		glColor3f(.6f, .1f, .51f);
		glBegin(GL_LINE_STRIP);
		for(j=0; j<11; j++) {
			p = spline[i].calculateBezierPoint(0.1f * j);
			glVertex3fv((GLfloat *)&p);
		}
		glEnd();
		
		spline[i].deCasteljauSplit(a, b);
		
		glColor3f(.6f, .71f, .1f);
		glBegin(GL_LINE_STRIP);
		for(j=0; j<11; j++) {
			p = a.calculateBezierPoint(0.1f * j);
			glVertex3fv((GLfloat *)&p);
		}
		glEnd();
		
		glColor3f(.1f, .71f, .1f);
		glBegin(GL_LINE_STRIP);
		for(j=0; j<11; j++) {
			p = b.calculateBezierPoint(0.1f * j);
			glVertex3fv((GLfloat *)&p);
		}
		glEnd();
	}
}

void BccWorld::testIntersection()
{
	BoundingBox box;
    box.setMin(m_testP.x - 1.f, m_testP.y - 1.f, m_testP.z - 1.f);
	box.setMax(m_testP.x + 1.f, m_testP.y + 1.f, m_testP.z + 1.f);
	
	bool intersected = 0;
	
	BezierSpline * spline = (BezierSpline *)m_splines->data();
    const unsigned ns = m_curve->numSegments();
    unsigned i;
    for(i=0; i < ns; i++) {
        intersected = testIntersection(spline[i], box);
		if(intersected) break;
    }
	
	intersected = m_curve->intersectBox(box);
	
	if(intersected) glColor3f(1.f, 0.f, 0.f);
	else glColor3f(0.f, 1.f, 0.f);
	
	m_drawer->boundingBox(box);
}

bool BccWorld::testIntersection(BezierSpline & spline, const BoundingBox & box)
{
	BoundingBox abox;
	abox.expandBy(spline.cv[0]);
	abox.expandBy(spline.cv[1]);
	abox.expandBy(spline.cv[2]);
	abox.expandBy(spline.cv[3]);
	
	if(!abox.intersect(box)) return false;
	
	if(abox.inside(box)) return true;
	
	BezierSpline stack[64];
	int stackSize = 2;
	spline.deCasteljauSplit(stack[0], stack[1]);
	
	while(stackSize > 0) {
		BezierSpline c = stack[stackSize - 1];
		stackSize--;
		
		abox.reset();
		abox.expandBy(c.cv[0]);
		abox.expandBy(c.cv[1]);
		abox.expandBy(c.cv[2]);
		abox.expandBy(c.cv[3]);
		
		m_drawer->boundingBox(abox);
		
		if(abox.inside(box)) return true;
		
		if(abox.intersect(box)) {
			if(abox.area() < 0.007f) return true;
			
			BezierSpline a, b;
			c.deCasteljauSplit(a, b);
			
			stack[ stackSize ] = a;
			stackSize++;
			stack[ stackSize ] = b;
			stackSize++;
		}
	}
	
	return false;
}
