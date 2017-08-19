/*
 *  PlantPiece.h
 *  garden
 *
 *  holds geometry, tm for branching
 *
 *  Created by jian zhang on 4/15/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_PLANT_PIECE_H
#define GAR_PLANT_PIECE_H

#include <math/Matrix44F.h>
#include <vector>

namespace aphid {
class ATriangleMesh;
class BoundingBox;

namespace cvx {
class Triangle;
}

namespace sdb {
template<typename T>
class VectorArray;

}

}

class PlantPiece {
/// local space parent
	aphid::Matrix44F m_tm;
	
typedef std::vector<PlantPiece * > ChildListTyp;
	ChildListTyp m_childPieces;
	PlantPiece * m_parentPiece;
	aphid::ATriangleMesh * m_geom;
	float m_exclR;
/// as instance id
	int m_geomid;
	
typedef aphid::cvx::Triangle GeomElmTyp;
typedef aphid::sdb::VectorArray<GeomElmTyp > GeomElmArrTyp;
	
public:
	PlantPiece(PlantPiece * parent = NULL);
	virtual ~PlantPiece();
/// add child piece	
	void addBranch(PlantPiece * c);
	
	void setTransformMatrix(const aphid::Matrix44F &tm);
	const aphid::Matrix44F & transformMatrix() const;
	
	int numBranches() const;
	const PlantPiece * branch(const int & i) const;
	
	void setGeometry(aphid::ATriangleMesh * geom, const int & geomId);
	const aphid::ATriangleMesh * geometry() const;
	
	void setExclR(const float & x);
/// composite exclusion radius of child 
	void setExclRByChild();
	const float & exclR() const;

	void countNumTms(int & count) const;
	void extractTms(aphid::Matrix44F * dst,
			int & count) const;
	void extractGeomIds(int * dst,
			int & count) const;
			
	void getGeom(GeomElmArrTyp * dst,
					aphid::BoundingBox & box,
					const aphid::Matrix44F & relTm);
					
	void worldTransformMatrix(aphid::Matrix44F & dst) const;
	
protected:

private:
	void getGeomElm(GeomElmArrTyp * dst,
					aphid::BoundingBox & box,
					const aphid::Matrix44F & relTm);
	
};

#endif