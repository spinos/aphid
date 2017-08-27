/*
 *  VegetationPatch.h
 *  garden
 *
 *  grow a patch via dart throwing around origin
 *
 *  Created by jian zhang on 4/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_VEGETATION_PATCH_H
#define GAR_VEGETATION_PATCH_H

#include <vector>
#include <ogl/DrawTriangle.h>

#include <ExampVox.h>

namespace aphid {

class Matrix44F;

}

class PlantPiece;

class VegetationPatch : public aphid::ExampVox, public aphid::DrawTriangle {

	typedef std::vector<PlantPiece *> PlantListTyp;
	typedef aphid::cvx::Triangle GeomElmTyp;
	typedef aphid::sdb::VectorArray<GeomElmTyp > GeomElmArrTyp;
/// to roots
	PlantListTyp m_plants;
	float m_tmv[16];
	float m_yardR;
	float m_tilt;
	
public:
	VegetationPatch();
	virtual ~VegetationPatch();
	
	int numPlants() const;
	const PlantPiece * plant(const int & i) const;
	
	void addPlant(PlantPiece * pl);
	void clearPlants();
	
	bool isFull() const;
	
	void setTilt(const float & x);
	const float & tilt() const;
	
	const float & yardRadius() const;
	
	void setTransformation(const aphid::Matrix44F & tm);

	const float * transformationV() const;
	
	int getNumTms();
	void extractTms(aphid::Matrix44F * dst);
	void extractGeomIds(int * dst);
	
	void getGeom(GeomElmArrTyp * dst,
					aphid::BoundingBox & box);
					
	void setTriangleDrawCache(const GeomElmArrTyp & src);
	
	virtual void voxelize3(aphid::sdb::VectorArray<aphid::cvx::Triangle> * tri,
							const aphid::BoundingBox & bbox);
	
protected:

private:
/// intersect with existing plants	
	bool intersectPlants(const aphid::Vector3F & pos, 
		const float & r) const;
};

#endif