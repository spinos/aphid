/*
 *  EbpMeshSample.cpp
 *  
 *
 *  Created by jian zhang on 7/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "EbpMeshSample.h"
#include <geom/PrimInd.h>
#include <geom/ATriangleMesh.h>

namespace aphid {

EbpMeshSample::EbpMeshSample()
{
}

EbpMeshSample::~EbpMeshSample()
{}

void EbpMeshSample::sample(ATriangleMesh* msh)
{	
	const Vector3F* ps = msh->points();
	std::vector<cvx::Triangle * > tris;
    sdb::Sequence<int> sels;
    
	const int nt = msh->numTriangles();
	for(int i=0;i<nt;++i) {
	    const unsigned * trii = msh->triangleIndices(i);
	    cvx::Triangle * ta = new cvx::Triangle;
	    ta->set(ps[trii[0]], ps[trii[1]], ps[trii[2]] );
	    tris.push_back(ta);
	    sels.insert(i);
	    
	}
	
	const BoundingBox bbx = msh->calculateGeomBBox();
	const float gz = bbx.radius() * .91f;
	
typedef PrimInd<sdb::Sequence<int>, std::vector<cvx::Triangle * >, cvx::Triangle > TIntersect;
	TIntersect fintersect(&sels, &tris);
	
#define GrdLvl 3
	EbpGrid::fillBox(fintersect, gz);
	EbpGrid::subdivideToLevel<TIntersect>(fintersect, 0, GrdLvl);
	EbpGrid::insertNodeAtLevel(GrdLvl);
	EbpGrid::cachePositions();

	for(int i=0;i<10;++i) {
		EbpGrid::updateFlat();    
	}	
}

int EbpMeshSample::numSamples()
{ return numCellsAtLevel(3); }

}
