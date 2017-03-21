/*
 *  LandBlock.h
 *  
 *  a single piece of land
 *
 *  Created by jian zhang on 3/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_LAND_BLOCK_H
#define APH_TTG_LAND_BLOCK_H

#include <sdb/Entity.h>
#include <math/Vector3F.h>
#include <vector>

namespace aphid {

class BoundingBox;

namespace img {

class HeightField;

}

class ATriangleMesh;

template<typename T1, typename T2>
class TetraGridTriangulation;

namespace ttg {

class HeightBccGrid;
class GlobalElevation;

template<typename T>
class GenericTetraGrid;

template<typename T>
class TetrahedronDistanceField;

class LandBlock : public sdb::Entity {

	Vector3F m_origin;
	int m_level;

public:
typedef HeightBccGrid BccTyp;
typedef	GenericTetraGrid<float > TetGridTyp;
typedef TetrahedronDistanceField<TetGridTyp > FieldTyp;
typedef TetraGridTriangulation<float, TetGridTyp > MesherTyp;

private:
	BccTyp * m_bccg;
	TetGridTyp * m_tetg;
	FieldTyp * m_field;
	MesherTyp * m_mesher;
	ATriangleMesh * m_frontMesh;
	img::HeightField * m_heightField;
	
public:
	LandBlock(sdb::Entity * parent = NULL);
	virtual ~LandBlock();
	
	void rebuild();
	
	void processHeightField();
	void triangulate();
	
	const TetGridTyp * grid() const;
	const FieldTyp * field() const;
	const ATriangleMesh * frontMesh() const;
	
protected:
	void toUV(float & u, float & v, const Vector3F & p) const;
	float heightSampleFilterSize() const;
	void getTouchedHeightFields(std::vector<int> & inds) const;
	void senseHeightField(Array3<float> & sigY,
							const img::HeightField & fld) const;
	
	BoundingBox buildBox() const;
	
private:	
};

}

}
#endif