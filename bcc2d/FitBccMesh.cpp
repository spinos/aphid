/*
 *  FitBccMesh.cpp
 *  testbcc
 *
 *  Created by jian zhang on 4/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "FitBccMesh.h"
#include <BccGlobal.h>
#include "FitBccMeshBuilder.h"
FitBccMesh::FitBccMesh() 
{
	m_anchors = new BaseBuffer;
}

FitBccMesh::~FitBccMesh() 
{
	delete m_anchors;
}

void FitBccMesh::create(GeometryArray * geoa, KdIntersection * anchorIntersect,
						float groupNCvRatio,
	           unsigned minNumGroups,
	           unsigned maxNumGroups)
{
	std::vector<Vector3F > tetrahedronP;
	std::vector<unsigned > tetrahedronInd;
	
	FitBccMeshBuilder builder;
	builder.build(geoa, tetrahedronP, tetrahedronInd, groupNCvRatio, minNumGroups, maxNumGroups);
	
	unsigned nt = tetrahedronInd.size()/4;
	unsigned np = tetrahedronP.size();
	
	std::cout<<" n tetrahedrons "<<nt<<"\n";
	std::cout<<" n vertices "<<np<<"\n";
	
	setNumPoints(np);
	setNumIndices(nt * 4);
	createBuffer(np, nt * 4);
	m_anchors->create(np * 4);
	
	unsigned i;
	for(i=0;i<np;i++) points()[i] = tetrahedronP[i];
	
	for(i=0;i<nt*4;i++) indices()[i] = tetrahedronInd[i];
	
	resetAnchors(np);
	addAnchors(anchorIntersect);
}

void FitBccMesh::resetAnchors(unsigned n)
{
	unsigned * anchor = (unsigned *)m_anchors->data();
	unsigned i=0;
	for(; i < n; i++)
		anchor[i] = 0;
}

void FitBccMesh::addAnchors(KdIntersection * anchorIntersect)
{
	unsigned * anchor = (unsigned *)m_anchors->data();
	
	Vector3F q[4];
	unsigned j, i=0;
	for(; i< numTetrahedrons(); i++) {
        unsigned * tet = &indices()[i*4];
        for(j=0; j< 4; j++)
            q[j] = points()[tet[j]]; 
        
		if(!anchorIntersect->intersectTetrahedron(q)) continue;
		
		for(j=0; j< 4; j++) {
			anchor[tet[j]] = 1;
		}
    }
}
