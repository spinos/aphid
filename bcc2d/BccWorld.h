#ifndef BCCWORLD_H
#define BCCWORLD_H
#include <ALLMath.h>
#include <BoundingBox.h>
#include <HTetrahedronMesh.h>
#include "BccGlobal.h"

class CurveGroup;
class KdTreeDrawer;
class BaseBuffer;
class KdCluster;
class KdIntersection;
class GeometryArray;
class APointCloud;
struct BezierSpline;
class BccMesh;
class FitBccMesh;
class ATriangleMesh;

class BccWorld {
public:
	BccWorld(KdTreeDrawer * drawer);
    virtual ~BccWorld();
    
    void draw();
    bool save();
    
private:
	bool createCurveGeometryFromFile();
	void createTestCurveGeometry();
	void createRandomCurveGeometry();
	void createCurveStartP();
	void createAnchorIntersect();
	void createTetrahedronMeshes();
	void createTriangleMeshesFromFile();
    bool readCurveDataFromFile();
	bool readTriangleDataFromFile();
	void drawCurveStars();

	void drawTetrahedronMesh();
	void drawTetrahedronMesh(unsigned nt, Vector3F * points, unsigned * indices);
	void drawAnchor();
	void drawTriangleMesh();
private:
    KdTreeDrawer * m_drawer;
    CurveGroup * m_curves;
	KdCluster * m_cluster;
	KdIntersection * m_anchorIntersect;
	GeometryArray * m_allGeo;
	APointCloud * m_curveStartP;
	GeometryArray * m_triangleMeshes;
#if WORLD_USE_FIT
	FitBccMesh * m_meshes;
#else
	BccMesh * m_meshes;
#endif
	unsigned m_numMeshes;
};

#endif        //  #ifndef BCCWORLD_H

