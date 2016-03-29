/*
 *  Forest.h
 *  proxyPaint
 *
 *  Created by jian zhang on 1/29/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "PlantSelection.h"
#include <Quaternion.h>
#include <Matrix44F.h>
#include <KdEngine.h>
#include <ConvexShape.h>
#include <ATriangleMesh.h>
#include <IntersectionContext.h>

/* http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
 * qw= √(1 + m00 + m11 + m22) /2
 * qx = (m21 - m12)/( 4 *qw)
 * qy = (m02 - m20)/( 4 *qw)
 * qz = (m10 - m01)/( 4 *qw)
 */
namespace aphid {

class ExampVox;

/*
 *  plant and ground data
 *
 */

class Forest {

	sdb::WorldGrid<sdb::Array<int, Plant>, Plant > * m_grid;
	std::vector<PlantData *> m_pool;
	std::vector<Plant *> m_plants;
    std::vector<ATriangleMesh *> m_grounds;
	std::vector<ExampVox *> m_examples;
	std::map<ExampVox *, unsigned> m_exampleIndices;
	KdNTree<cvx::Triangle, KdNode4 > * m_ground;
	sdb::VectorArray<cvx::Triangle> m_triangles;
	PlantSelection * m_activePlants;
	IntersectionContext m_intersectCtx;
	Geometry::ClosestToPointTestResult m_closestPointTest;
	SphereSelectionContext * m_selectCtx;
	unsigned m_numPlants;
	
public:
	Forest();
	virtual ~Forest();
	
    void setSelectionRadius(float x);
	unsigned numActiveGroundFaces();
	const unsigned & numActivePlants() const;
	void removeAllPlants();
	const float & selectionRadius() const;
    
protected:
	void resetGrid(float gridSize);
	void updateGrid();
	void updateNumPlants();
	void clearGroundMeshes();
    void setGroundMesh(ATriangleMesh * trimesh, unsigned idx);
    void buildGround();
    void setSelectTypeFilter(int flt);
    bool selectPlants(const Ray & ray, SelectionContext::SelectMode mode);
	bool selectGroundFaces(const Ray & ray, SelectionContext::SelectMode mode);
	
	unsigned numCells();
	unsigned numGroundMeshes() const;
    unsigned numPlants() const;
	const BoundingBox & gridBoundingBox() const;
	sdb::WorldGrid<sdb::Array<int, Plant>, Plant > * grid();
	sdb::Array<int, PlantInstance> * activePlants();
	PlantSelection * selection();
	KdNTree<cvx::Triangle, KdNode4 > * ground();
	const KdNTree<cvx::Triangle, KdNode4 > * ground() const;
	IntersectionContext * intersection();
	SphereSelectionContext * activeGround();
	ATriangleMesh * getGroundMesh(unsigned idx) const;
	const std::vector<ATriangleMesh *> & groundMeshes() const;
	
	const float & plantSize(int idx) const;
	
	void displacePlantInGrid(PlantInstance * inst );
	void bindToGround(PlantData * plantd, const Vector3F & origin, Vector3F & dest);
	bool getBindPoint(Vector3F & pos, GroundBind * bind);
	
	bool closeToOccupiedPosition(const Vector3F & pos, 
					const float & minDistance);
	bool intersectGround(const Ray & ray);
	void addPlant(const Matrix44F & tm,
					const GroundBind & bind,
					const int & plantTypeId);
    const Vector3F & selectionCenter() const;
    const Vector3F & selectionNormal() const;
    
    bool isGroundEmpty() const;
    void addPlantExample(ExampVox * x);
	ExampVox * plantExample(unsigned idx);
	const ExampVox * plantExample(unsigned idx) const;
	std::string groundBuildLog() const;
	
	const sdb::VectorArray<cvx::Triangle> & triangles() const;
	
private:
	bool testNeighborsInCell(const Vector3F & pos, 
					const float & minDistance,
					sdb::Array<int, Plant> * cell);
	
};

}