#ifndef BCCWORLD_H
#define BCCWORLD_H
#include <ALLMath.h>
class BezierCurve;
class GeoDrawer;
class BccGrid;
struct SimpleBezierSpline;
class BccWorld {
public:
    BccWorld(GeoDrawer * drawer);
    virtual ~BccWorld();
    
    void draw();
    
    void moveTestP(float x, float y, float z);
    
private:
    void testSegments();
    void testDistanctToCurve();
    void testDistanceToPoint(SimpleBezierSpline & spline, const Vector3F & pnt, float & minDistance, Vector3F & closestP);
private:
    Vector3F m_testP;
    GeoDrawer * m_drawer;
    BezierCurve * m_curve;
    BezierCurve * m_segmentCurve;
    BccGrid * m_grid;
};

#endif        //  #ifndef BCCWORLD_H

