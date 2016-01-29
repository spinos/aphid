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
#include <KdTree.h>
#include <ATriangleMesh.h>

/* http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
 * qw= √(1 + m00 + m11 + m22) /2
 * qx = (m21 - m12)/( 4 *qw)
 * qy = (m02 - m20)/( 4 *qw)
 * qz = (m10 - m01)/( 4 *qw)
 */

namespace sdb {
/// (plant id, (world orientation, world position, triangle bind id) )
typedef Triple<Quaternion, Vector3F, int > RotPosTri;
class Plant : public Pair<int, RotPosTri>
{
public:
	
	const bool operator==(const Plant & another) const {
		return index == another.index;
	}
	
};

class Forest {

	WorldGrid<Array<int, Plant>, Plant > * m_grid;
	std::vector<RotPosTri *> m_pool;
	std::vector<Plant *> m_plants;
	KdTree * m_ground;
	
public:
	Forest();
	virtual ~Forest();
	
	void resetGrid(float gridSize);
	void finishGrid();
	void addPlant(const Quaternion & orientation, 
					const Vector3F & position,
					const int & triangleId);
					
	void resetGround();
	void addGroundMesh(ATriangleMesh * trimesh);
	void finishGround();
	
protected:
	const BoundingBox boundingBox() const;
	unsigned numPlants() const;
	
private:

};

}