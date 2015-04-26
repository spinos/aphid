#ifndef BCCWORLD_H
#define BCCWORLD_H
#include <ALLMath.h>
#include <BoundingBox.h>
#include <HTetrahedronMesh.h>

class CurveGroup;
class KdTreeDrawer;
class BaseBuffer;
class KdCluster;
class KdIntersection;
class GeometryArray;
class APointCloud;
struct BezierSpline;
class BccMesh;

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
	void createCurveStartP();
	void createAnchorIntersect();
	void createMeshes();
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
    BaseBuffer * m_splineBuf;
	BaseBuffer * m_curveStartBuf;
	CurveGroup * m_curves;
	KdCluster * m_cluster;
	KdIntersection * m_anchorIntersect;
	GeometryArray * m_allGeo;
	APointCloud * m_curveStartP;
	unsigned m_numSplines;
	TetrahedronMeshData m_mesh;
	BccMesh * m_meshes;
	unsigned m_numMeshes;
};

#endif        //  #ifndef BCCWORLD_H

