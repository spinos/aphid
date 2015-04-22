#ifndef BCCWORLD_H
#define BCCWORLD_H
#include <ALLMath.h>
#include <BoundingBox.h>
#include <HTetrahedronMesh.h>
class CurveGroup;
class GeoDrawer;
class BccGrid;
class BaseBuffer;
struct BezierSpline;

class BccWorld {
public:
	BccWorld(GeoDrawer * drawer);
    virtual ~BccWorld();
    
    void draw();
    bool save();
    void moveTestP(float x, float y, float z);
    
private:
	void createTestCurves();
    void testDistanctToCurve();
    void testDistanceToPoint(BezierSpline & spline, const Vector3F & pnt, float & minDistance, Vector3F & closestP);
	void testSpline();
	void testIntersection();
	bool testIntersection(BezierSpline & spline, const BoundingBox & box);
	void testTetrahedronBoxIntersection();
	bool intersectTetrahedron(Vector3F * p);
	bool intersectTetrahedron(BezierSpline & spline, Vector3F * p);
	void testLineLine();
	void testVicinity();
	void drawCurves();
	bool readCurvesFromFile();
	void drawCurveStars();
	void resetAnchors(unsigned n);
	void createMeshData(unsigned nt, unsigned nv);
	void drawMesh();
	void drawMesh(unsigned nt, Vector3F * points, unsigned * indices);
	void drawAnchor();
private:
    Vector3F m_testP;
    GeoDrawer * m_drawer;
    BccGrid * m_grid;
	BaseBuffer * m_splineBuf;
	BaseBuffer * m_curveStartBuf;
	CurveGroup * m_curves;
	unsigned m_numSplines;
	TetrahedronMeshData m_mesh;
};

#endif        //  #ifndef BCCWORLD_H

