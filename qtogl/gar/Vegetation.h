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

#include <map>

namespace aphid {
class ATriangleMesh;
}

class VegetationPatch;

class Vegetation {

public:
	typedef aphid::ATriangleMesh * GeomPtrTyp;
	
private:
	std::map<int, GeomPtrTyp > m_cachedGeom;
	std::map<int, GeomPtrTyp >::iterator m_geomIter;
	
#define NUM_ANGLE 11
#define NUM_VARIA 11
#define TOTAL_NUM_P 121
	VegetationPatch * m_patches[TOTAL_NUM_P];
	int m_numPatches;
	
public:
	Vegetation();
	virtual ~Vegetation();
	
	VegetationPatch * patch(const int & i);
	
	void setNumPatches(int x);
	const int & numPatches() const;
	
	int getMaxNumPatches() const;
	
	void rearrange();
	int getNumInstances();
	
	aphid::ATriangleMesh * findGeom(const int & k);
	void addGeom(const int & k, aphid::ATriangleMesh * v);
	
	int numCachedGeoms() const;
	
	void geomBegin(std::string & mshName, GeomPtrTyp & mshVal);
	void geomNext(std::string & mshName, GeomPtrTyp & mshVal);
/// for each patch, sample grid
	void voxelize();
	
protected:

private:
	void clearCachedGeom();
	std::string getGeomName(const int & k);
	void voxelize(VegetationPatch * ap);
	
};

#endif