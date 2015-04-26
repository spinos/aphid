#ifndef BCCWORLD_H
#define BCCWORLD_H
#include <ALLMath.h>
#include <BoundingBox.h>
#include <HTetrahedronMesh.h>
class CurveGroup;
class KdTreeDrawer;
class BccGrid;
class BaseBuffer;
class KdCluster;
class KdIntersection;
class GeometryArray;
class APointCloud;
struct BezierSpline;

class BccWorld {
public:
	BccWorld(KdTreeDrawer * drawer);
    virtual ~BccWorld();
    
    void draw();
    bool save();
    void moveTestP(float x, float y, float z);
    
private:
	void createTestCurveData();
	void createRandomCurveGeometry();
	void createCurveGeometry();
	void createGroupIntersection();
	void createCurveStartP();
	void createAnchorIntersect();
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
	bool readCurveDataFromFile();
	void drawCurveStars();
	void resetAnchors(unsigned n);
	void createMeshData(unsigned nt, unsigned nv);
	void drawMesh();
	void drawMesh(unsigned nt, Vector3F * points, unsigned * indices);
	void drawAnchor();
private:
    Vector3F m_testP;
    KdTreeDrawer * m_drawer;
    BccGrid * m_grid;
	BaseBuffer * m_splineBuf;
	BaseBuffer * m_curveStartBuf;
	CurveGroup * m_curves;
	KdCluster * m_cluster;
	KdIntersection * m_intersect;
	KdIntersection * m_anchorIntersect;
	GeometryArray * m_allGeo;
	APointCloud * m_curveStartP;
	unsigned m_numSplines;
	TetrahedronMeshData m_mesh;
};

#endif        //  #ifndef BCCWORLD_H

