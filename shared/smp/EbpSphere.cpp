/*
 *  EbpSphere.cpp
 *  
 *
 *  Created by jian zhang on 7/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "EbpSphere.h"
#include <geom/PrimInd.h>
#include <geom/GeodesicSphereMesh.h>

namespace aphid {

EbpSphere::EbpSphere()
{
	TriangleGeodesicSphere geodsph(6);
	const int nv = geodsph.numPoints();
	Vector3F * sphps = geodsph.points();
	for(int i=0;i<nv;++i) {
	    sphps[i] *= 9.f;
	}
	
	std::vector<cvx::Triangle * > tris;
    sdb::Sequence<int> sels;
    
	const int nt = geodsph.numTriangles();
	for(int i=0;i<nt;++i) {
	    const unsigned * trii = geodsph.triangleIndices(i);
	    cvx::Triangle * ta = new cvx::Triangle;
	    ta->set(sphps[trii[0]], sphps[trii[1]], sphps[trii[2]] );
	    tris.push_back(ta);
	    sels.insert(i);
	    
	}
	
typedef PrimInd<sdb::Sequence<int>, std::vector<cvx::Triangle * >, cvx::Triangle > TIntersect;
	TIntersect fintersect(&sels, &tris);
	
#define GrdLvl 3
	EbpGrid::fillBox(fintersect, 9.3f);
	EbpGrid::subdivideToLevel<TIntersect>(fintersect, 0, GrdLvl);
	EbpGrid::insertNodeAtLevel(GrdLvl);
	EbpGrid::cachePositions();

	for(int i=0;i<10;++i) {
		EbpGrid::updateNormalized(10.f);    
	}	
}

EbpSphere::~EbpSphere()
{}

int EbpSphere::numSamples()
{ return numCellsAtLevel(3); }

}