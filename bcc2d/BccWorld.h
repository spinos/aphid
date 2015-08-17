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
class CurveReduction;
class ATetrahedronMeshGroup;

class BccWorld {
public:
	BccWorld(KdTreeDrawer * drawer);
    virtual ~BccWorld();
    
    void draw();
    bool save();
	
	void select(const Ray * r);
	void clearSelection();
    void reduceSelected(float x);
    void rebuildTetrahedronsMesh(float deltaNumGroups);
    const float totalCurveLength() const;
    const unsigned numCurves() const;
	const unsigned numTetrahedrons() const;
	const unsigned numPoints() const;
private:
	bool createCurveGeometryFromFile();
	void createTestCurveGeometry();
	void createRandomCurveGeometry();
	void createTetrahedronMeshes();
	void createTriangleMeshesFromFile();
    bool readCurveDataFromFile();
	bool readTriangleDataFromFile();
	void drawCurveStars();

	void drawTetrahedronMesh();
	void drawTetrahedronMesh(unsigned nt, Vector3F * points, unsigned * indices);
	void drawAnchor();
	void drawTriangleMesh();
    void clearTetrahedronMesh();
	void reduceAllGroups();
	void reduceGroup(unsigned igroup);
	float groupCurveLength(GeometryArray * geos);
	void rebuildGroupTetrahedronMesh(unsigned igroup, GeometryArray * geos);
	ATetrahedronMeshGroup * combinedTetrahedronMesh();
private:
    KdTreeDrawer * m_drawer;
    CurveGroup * m_curves;
	KdCluster * m_cluster;
	KdIntersection * m_triIntersect;
	GeometryArray * m_allGeo;
	GeometryArray * m_triangleMeshes;
#if WORLD_USE_FIT
	FitBccMesh * m_meshes;
#else
	BccMesh * m_meshes;
#endif
    CurveReduction * m_reducer;
	unsigned m_numMeshes, m_numCurves, m_totalNumTetrahedrons, m_totalNumPoints;
    float m_totalCurveLength;
    float m_estimatedNumGroups;
};

#endif        //  #ifndef BCCWORLD_H

