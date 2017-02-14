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
#include <kd/KdEngine.h>
#include <math/RayMarch.h>
#include <PlantSelection.h>

namespace aphid {

class ForestCell;
class ExampVox;
class ATriangleMesh;
struct Float2;

class Forest {

	sdb::WorldGrid<ForestCell, Plant > * m_grid;
	std::vector<PlantData *> m_pool;
	std::vector<Plant *> m_plants;
    std::vector<ATriangleMesh *> m_grounds;
	std::map<int, ExampVox *> m_examples;
	std::map<ExampVox *, unsigned> m_exampleIndices;
	KdNTree<cvx::Triangle, KdNode4 > * m_ground;
	sdb::VectorArray<cvx::Triangle> m_triangles;
	PlantSelection * m_activePlants;
	IntersectionContext m_intersectCtx;
	ClosestToPointTestResult m_closestPointTest;
	SphereSelectionContext * m_selectCtx;
	RayMarch m_march;
	unsigned m_numPlants;
	int m_lastPlantInd;
	
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
	sdb::WorldGrid<ForestCell, Plant > * grid();
	PlantSelection * selection();
	PlantSelection::SelectionTyp * activePlants();
	KdNTree<cvx::Triangle, KdNode4 > * ground();
	const KdNTree<cvx::Triangle, KdNode4 > * ground() const;
	IntersectionContext * intersection();
	SphereSelectionContext * activeGround();
	ATriangleMesh * getGroundMesh(const int & idx) const;
	const std::vector<ATriangleMesh *> & groundMeshes() const;
	
	const float & plantSize(const int & idx);
	
	void displacePlantInGrid(PlantInstance * inst );
	bool bindToGround(GroundBind * bind, const Vector3F & origin, Vector3F & dest);
	void bindToGround(PlantData * plantd, const Vector3F & origin, Vector3F & dest);
	void setBind(GroundBind * bind) const;
/// ground normal at bind point
	Vector3F bindNormal(const GroundBind * bind) const;
/// -1: disable binding
///  0: failed, rebind
///  1: success
	int getBindPoint(Vector3F & pos, GroundBind * bind);
	
	struct CollisionContext {
		Vector3F _pos;
		int _bundleIndex;
		int _minIndex;
		float _minDistance;
		float _maxDistance;
		float _bundleScaling;
		float _radius;
		
		float getXMin() {
			return _pos.x - _radius;
		}
		
		float getXMax() {
			return _pos.x + _radius;
		}
		
		float getYMin() {
			return _pos.y - _radius;
		}
		
		float getYMax() {
			return _pos.y + _radius;
		}
		
		float getZMin() {
			return _pos.z - _radius;
		}
		
		float getZMax() {
			return _pos.z + _radius;
		}
		
		bool contact(const Vector3F & q,
					const float & r) {
			return _pos.distanceTo(q) < (_radius + _minDistance + r);
		}
		
	};
	
	bool closeToOccupiedPosition(CollisionContext * ctx);
	bool closeToOccupiedBundlePosition(CollisionContext * ctx);
	bool intersectGround(const Ray & ray);
	bool intersectGrid(const Ray & ray);
	void addPlant(const Matrix44F & tm,
					const GroundBind & bind,
					const int & plantTypeId);
    const Vector3F & selectionCenter() const;
    const Vector3F & selectionNormal() const;
    
    bool isGroundEmpty() const;
    void addPlantExample(ExampVox * x, const int & islot);
	ExampVox * plantExample(const int & idx);
	
	std::string groundBuildLog() const;
	
	const sdb::VectorArray<cvx::Triangle> & triangles() const;
	int numPlantExamples() const;
	bool closestPointOnGround(Vector3F & dest,
					const Vector3F & origin,
					const float & maxDistance);
		
	void onPlantChanged();
	void intersectWorldBox(const Ray & ray);
	const int & lastPlantIndex() const;

	void getBindTexcoord(Float2 & dst) const;
	
private:
	bool testNeighborsInCell(CollisionContext * ctx,
					ForestCell * cell);
	
};

}
#endif