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
	BccWorld();
    virtual ~BccWorld();
	
	void addTriangleMesh(ATriangleMesh * m);
	void addCurveGroup(CurveGroup * m);
	bool buildTetrahedronMesh();
    
	void select(const Ray * r);
	void clearSelection();
    void reduceSelected(float x);
    void rebuildTetrahedronsMesh(float deltaNumGroups);
    const float totalCurveLength() const;
    const unsigned numCurves() const;
	const unsigned numTetrahedrons() const;
	const unsigned numPoints() const;
	unsigned numTetrahedronMeshes() const;
	ATetrahedronMesh * tetrahedronMesh(unsigned i) const;
	GeometryArray * triangleGeometries() const;
	GeometryArray * selectedGroup(unsigned & idx) const;
	float drawAnchorSize() const;
	ATetrahedronMeshGroup * combinedTetrahedronMesh();
	
private:
	bool createAllCurveGeometry();
	void createCurveGeometry(unsigned geoBegin, CurveGroup * data);
	bool createTriangleGeometry();
	void createTetrahedronMeshes();
    
	void clearTetrahedronMesh();
	void reduceAllGroups();
	void reduceGroup(unsigned igroup);
	float groupCurveLength(GeometryArray * geos);
	void rebuildGroupTetrahedronMesh(unsigned igroup, GeometryArray * geos);
	
private:
	KdCluster * m_cluster;
	KdIntersection * m_triIntersect;
	GeometryArray * m_allGeo;
	GeometryArray * m_triangleMeshes;
#if WORLD_USE_FIT
	FitBccMesh * m_meshes;
#else
	BccMesh * m_meshes;
#endif
	std::vector<ATriangleMesh *> m_triGeos;
	std::vector<CurveGroup *> m_curveGeos;
    CurveReduction * m_reducer;
	unsigned m_numMeshes, m_numCurves, m_totalNumTetrahedrons, m_totalNumPoints;
    float m_totalCurveLength;
    float m_estimatedNumGroups;
};

#endif        //  #ifndef BCCWORLD_H

