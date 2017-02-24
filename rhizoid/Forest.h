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
#include <string>
#include <map>

namespace aphid {

class ForestCell;
class ForestGrid;
class ExampVox;
class ATriangleMesh;
class ANoise3Sampler;
class SampleFilter;
class ExrImage;
class AFrustum;
struct Float2;

class Forest {

	ForestGrid * m_grid;
	std::vector<PlantData *> m_pool;
	std::vector<Plant *> m_plants;
    std::vector<ATriangleMesh *> m_groundMeshes;
	std::map<int, ExampVox *> m_examples;
	std::map<ExampVox *, unsigned> m_exampleIndices;
	KdNTree<cvx::Triangle, KdNode4 > * m_ground;
	sdb::VectorArray<cvx::Triangle> m_triangles;
	PlantSelection * m_activePlants;
	IntersectionContext m_intersectCtx;
	ClosestToPointTestResult m_closestPointTest;
	SampleFilter * m_sampleFlt;
	RayMarch m_march;
	int m_numPlants;
	int m_lastPlantInd;
	
public:
	Forest();
	virtual ~Forest();
	
    const int & numActivePlants() const;
	const float & selectionRadius() const;
	const float & gridSize() const;
	const float & filterPortion() const;
	int numVisibleSamples();
	
	void getStatistics(std::map<std::string, std::string > & stats);
	
protected:
    void setSelectionRadius(float x);
	void setSelectionFalloff(float x);
	void resetGrid(float x);
	void updateGrid();
	void countNumSamples();
	void countNumPlants();
	void clearGroundMeshes();
    void setGroundMesh(ATriangleMesh * trimesh, unsigned idx);
    void buildGround();
    void setSelectTypeFilter(int flt);
	bool selectTypedPlants(int x);
    bool selectPlants(const Ray & ray, SelectionContext::SelectMode mode);
	bool selectGroundSamples(const Ray & ray, SelectionContext::SelectMode mode);
	bool selectGroundSamples(const AFrustum & fru, SelectionContext::SelectMode mode);
	
	unsigned numCells();
	unsigned numGroundMeshes() const;
    const int & numPlants() const;
	const BoundingBox & gridBoundingBox() const;
	ForestGrid * grid();
	PlantSelection * selection();
	PlantSelection::SelectionTyp * activePlants();
	KdNTree<cvx::Triangle, KdNode4 > * ground();
	const KdNTree<cvx::Triangle, KdNode4 > * ground() const;
	IntersectionContext * intersection();
	ATriangleMesh * getGroundMesh(const int & idx) const;
	const std::vector<ATriangleMesh *> & groundMeshes() const;
	
	const float & plantSize(const int & idx);
	
	void displacePlantInGrid(PlantInstance * inst );
	bool bindToGround(GroundBind * bind, const Vector3F & origin, Vector3F & dest);
	void bindToGround(PlantData * plantd, const Vector3F & origin, Vector3F & dest);
	void getClosestBind(GroundBind * bind) const;
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
		
	void onSampleChanged();
	void onPlantChanged();
	void intersectWorldBox(const Ray & ray);
	const int & lastPlantIndex() const;

	void getBindTexcoord(Float2 & dst) const;
	
	const int & sampleLevel() const;
	void processSampleFilter();
	
	void setFilterPortion(const float & x);
	void reshuffleSamples();
	void deselectSamples();
	void setFilterNoise(const ANoise3Sampler & param);
    void setFilterImage(const ExrImage * img);
    
	void clearAllPlants();
	
private:
	bool testNeighborsInCell(CollisionContext * ctx,
					ForestCell * cell);
	
};

}
#endif