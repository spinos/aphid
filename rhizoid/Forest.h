/*
 *  Forest.h
 *  proxyPaint
 *
 *  Created by jian zhang on 1/29/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <WorldGrid.h>
#include <Array.h>
#include <Quaternion.h>
#include <Matrix44F.h>
#include <KdTree.h>
#include <ATriangleMesh.h>
#include <IntersectionContext.h>
#include <SelectionContext.h>
#include <PseudoNoise.h>

/* http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
 * qw= √(1 + m00 + m11 + m22) /2
 * qx = (m21 - m12)/( 4 *qw)
 * qy = (m02 - m20)/( 4 *qw)
 * qz = (m10 - m01)/( 4 *qw)
 */
 
class TriangleRaster;

namespace sdb {
/// (plant id, (transformation, plant type id, triangle bind id) )
typedef Triple<Matrix44F, int, int > PlantData;
class Plant : public Pair<int, PlantData>
{
public:
	
	const bool operator==(const Plant & another) const {
		return index == another.index;
	}
	
};

/*
 *  plant and ground data
 *
 */

class Forest {

	WorldGrid<Array<int, Plant>, Plant > * m_grid;
	std::vector<PlantData *> m_pool;
	std::vector<Plant *> m_plants;
    std::vector<ATriangleMesh *> m_grounds;
	KdTree * m_ground;
	IntersectionContext m_intersectCtx;
	SelectionContext m_selectCtx;
	PseudoNoise m_pnoise;
	int m_seed;
	unsigned m_numPlants;
	
public:
	struct GrowOption {
		Vector3F m_upDirection;
		int m_plantId;
		float m_minScale, m_maxScale;
		float m_marginSize;
		float m_rotateNoise;
		bool m_alongNormal;
	};
	
	Forest();
	virtual ~Forest();
	
	unsigned numActiveGroundFaces();
	
protected:
	void resetGrid(float gridSize);
	void updateGrid();
	void addPlant(const Matrix44F & tm,
					const int & plantTypeId,
					const int & triangleId);
	const BoundingBox & gridBoundingBox() const;
	unsigned numPlants() const;
	void updateNumPlants();
	unsigned numCells();
	unsigned numGroundMeshes() const;
    void clearGroundMeshes();
    void setGroundMesh(ATriangleMesh * trimesh, unsigned idx);
    ATriangleMesh * getGroundMesh(unsigned idx) const;
    void buildGround();
    bool selectGroundFaces(const Ray & ray, SelectionContext::SelectMode mode);
	SelectionContext * activeGround();
	void growOnGround(GrowOption & option);
	
	virtual float plantSize(int idx) const;
	
	WorldGrid<Array<int, Plant>, Plant > * grid();
	
private:
	void growOnFaces(Geometry * geo, Sequence<unsigned> * components, 
					GrowOption & option);
	void growOnTriangle(TriangleRaster * tri, GrowOption & option);
	bool closeToOccupiedPosition(const Vector3F & pos, 
					const float & minDistance);
	Matrix44F randomSpaceAt(const Vector3F & pos, const GrowOption & option);
	
};

}