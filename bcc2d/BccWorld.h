#ifndef BCCWORLD_H
#define BCCWORLD_H
#include <ALLMath.h>
#include <BoundingBox.h>
class BezierCurve;
class GeoDrawer;
class BccGrid;
class BaseBuffer;
struct BezierSpline;

class BccWorld {
public:
    BccWorld(GeoDrawer * drawer);
    virtual ~BccWorld();
    
    void draw();
    
    void moveTestP(float x, float y, float z);
    
private:
    void testDistanctToCurve();
    void testDistanceToPoint(BezierSpline & spline, const Vector3F & pnt, float & minDistance, Vector3F & closestP);
	void testSpline();
	void testIntersection();
	bool testIntersection(BezierSpline & spline, const BoundingBox & box);
private:
    Vector3F m_testP;
    GeoDrawer * m_drawer;
    BezierCurve * m_curve;
    BccGrid * m_grid;
	BaseBuffer * m_splines;
};

#endif        //  #ifndef BCCWORLD_H

