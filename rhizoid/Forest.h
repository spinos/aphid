/*
 *  Forest.h
 *  proxyPaint
 *
 *  Created by jian zhang on 1/29/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_FOREST_H
#define APH_FOREST_H
#include <PlantCommon.h>
#include <KdEngine.h>
#include <ConvexShape.h>
#include <IntersectionContext.h>
#include <RayMarch.h>

namespace aphid {

class ExampVox;
class ATriangleMesh;
class PlantSelection;

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
	RayMarch m_march;
	unsigned m_numPlants;
	
public:
	Forest();
	virtual ~Forest();
	
    void setSelectionRadius(float x);
	unsigned numActiveGroundFaces();
	const int & numActivePlants() const;
	void removeAllPlants();
	const float & selectionRadius() const;
	const float & gridSize() const;
    
protected:
	void resetGrid(float x);
	void updateGrid();
	void updateNumPlants();
	void clearGroundMeshes();
    void setGroundMesh(ATriangleMesh * trimesh, unsigned idx);
    void buildGround();
    void setSelectTypeFilter(int flt);
	bool selectTypedPlants(int x);
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
	bool bindToGround(GroundBind * bind, const Vector3F & origin, Vector3F & dest);
	void bindToGround(PlantData * plantd, const Vector3F & origin, Vector3F & dest);
	void setBind(GroundBind * bind) const;
/// -1: disable binding
///  0: failed, rebind
///  1: success
	int getBindPoint(Vector3F & pos, GroundBind * bind);
	
	bool closeToOccupiedPosition(const Vector3F & pos, 
					const float & minDistance);
	bool intersectGround(const Ray & ray);
	bool intersectGrid(const Ray & ray);
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
	int numPlantExamples() const;
	bool closestPointOnGround(Vector3F & dest,
					const Vector3F & origin,
					const float & maxDistance);
		
	void onPlantChanged();
	void intersectWorldBox(const Ray & ray);
	
private:
	bool testNeighborsInCell(const Vector3F & pos, 
					const float & minDistance,
					sdb::Array<int, Plant> * cell);
	
};

}
#endif