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

namespace aphid {

class ATriangleMesh;

template<typename T1, typename T2>
class TetraGridTriangulation;

namespace ttg {

class AdaptiveBccGrid3;
class GlobalHeightField;

template<typename T>
class GenericTetraGrid;

template<typename T>
class TetrahedronDistanceField;

class LandBlock : public sdb::Entity {

	Vector3F m_origin;

public:
typedef	GenericTetraGrid<float > TetGridTyp;
typedef TetrahedronDistanceField<TetGridTyp > FieldTyp;
typedef TetraGridTriangulation<float, TetGridTyp > MesherTyp;

private:
	AdaptiveBccGrid3 * m_bccg;
	TetGridTyp * m_tetg;
	FieldTyp * m_field;
	MesherTyp * m_mesher;
	ATriangleMesh * m_frontMesh;
	
public:
	LandBlock(sdb::Entity * parent = NULL);
	virtual ~LandBlock();
	
	void processHeightField(const GlobalHeightField * elevation);
	void triangulate();
	
	const TetGridTyp * grid() const;
	const FieldTyp * field() const;
	const ATriangleMesh * frontMesh() const;
	
protected:
	
private:	
};

}

}
#endif