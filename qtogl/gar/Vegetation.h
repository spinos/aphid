/*
 *  Vegetation.h
 *  garden
 *
 *  collection of patches at different angles
 *
 *  Created by jian zhang on 4/26/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_VEGETATION_H
#define GAR_VEGETATION_H

#include <Variform.h>
#include <map>
#include <math/BoundingBox.h>

namespace aphid {
class ATriangleMesh;
}

class VegetationPatch;

class Vegetation : public aphid::Variform {
/// over all patches
	aphid::BoundingBox m_bbox;
	
public:
	typedef aphid::ATriangleMesh * GeomPtrTyp;
	
private:
	std::map<int, GeomPtrTyp > m_cachedGeom;
	std::map<int, GeomPtrTyp >::iterator m_geomIter;
	int m_curGeomId;
	
#define TOTAL_NUM_PAC 64
	VegetationPatch * m_patches[TOTAL_NUM_PAC];
	int m_numPatches;
	
public:
	Vegetation();
	virtual ~Vegetation();
	
	void setSynthByAngleAlign();
	void setSynthByRandom();
	
	VegetationPatch * patch(const int & i);
	
	void setNumPatches(int x);
	const int & numPatches() const;
	
	int getMaxNumPatches() const;
	
	void rearrange();
	int getNumInstances();
	
	aphid::ATriangleMesh * findGeom(const int & k);
	void addGeom(const int & k, aphid::ATriangleMesh * v);
/// as instance id
	int getGeomInd(aphid::ATriangleMesh * x);
	
	int numCachedGeoms() const;
	
	void geomBegin(std::string & mshName, GeomPtrTyp & mshVal);
	void geomNext(std::string & mshName, GeomPtrTyp & mshVal);
/// for each patch, sample grid
	void voxelize();
	
	const aphid::BoundingBox & bbox() const;
/// map only
	void clearGeom();
	
protected:

private:
	void clearCachedGeom();
/// compose name of current geom
	std::string getGeomName(const int & k);
	void voxelize(VegetationPatch * ap);
	
};

#endif