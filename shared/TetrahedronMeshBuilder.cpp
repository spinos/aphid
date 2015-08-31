/*
 *  TetrahedronMeshBuilder.cpp
 *  bcc
 *
 *  Created by jian zhang on 8/26/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "TetrahedronMeshBuilder.h"

float TetrahedronMeshBuilder::EstimatedGroupSize = 1.f;

TetrahedronMeshBuilder::TetrahedronMeshBuilder() {}
TetrahedronMeshBuilder::~TetrahedronMeshBuilder() {}

void TetrahedronMeshBuilder::build(GeometryArray * geos, 
					unsigned & ntet, unsigned & nvert, unsigned & nstripes) {}
					
void TetrahedronMeshBuilder::addAnchors(ATetrahedronMesh * mesh, unsigned n, KdIntersection * anchorMesh) {}
					
void TetrahedronMeshBuilder::getResult(ATetrahedronMeshGroup * mesh)
{
	const unsigned ntet = mesh->numTetrahedrons();
	const unsigned nvert = mesh->numPoints();
	const unsigned nstripes = mesh->numStripes();
	unsigned i;
	for(i=0;i<nvert;i++) mesh->points()[i] = tetrahedronP[i];
	for(i=0;i<ntet*4;i++) mesh->indices()[i] = tetrahedronInd[i];
	for(i=0;i<nstripes;i++) mesh->pointDrifts()[i] = pointDrifts[i];
	for(i=0;i<nstripes;i++) mesh->indexDrifts()[i] = indexDrifts[i];
}

void TetrahedronMeshBuilder::addAnchor(ATetrahedronMesh * mesh, 
					unsigned istripe,
					const Vector3F & p, unsigned tri)
{
	unsigned tetMax = mesh->numTetrahedrons() * 4;
	if(istripe < indexDrifts.size() -1)
		tetMax = indexDrifts[istripe + 1];
		
	BoundingBox box;
	unsigned i = indexDrifts[istripe];
	unsigned j;
	for(;i<tetMax;i+=4) {
		unsigned * tet = mesh->tetrahedronIndices(i/4);
		box.reset();
        for(j=0; j< 4; j++)
            box.expandBy(mesh->points()[tet[j]], 1e-3f); 
			
		if(box.center().distanceTo(p) > box.radius())
			continue;
		
		for(j=0; j< 4; j++)
			mesh->anchors()[tet[j]] = (1<<24 | tri);
	}
}

void TetrahedronMeshBuilder::addAnchor(ATetrahedronMesh * mesh, 
					KdIntersection * anchorMesh)
{
	const unsigned n = mesh->numTetrahedrons();
	Vector3F p[4];
	unsigned tri, i, j;
	for(i=0;i<n;i++) {
		unsigned * tet = mesh->tetrahedronIndices(i);
		for(j=0; j< 4; j++) p[j] = mesh->points()[tet[j]];
		
		if(!anchorMesh->intersectTetrahedron(p)) continue;
			
		tri = anchorMesh->intersectedElement();
		for(j=0; j< 4; j++) mesh->anchors()[tet[j]] = (1<<24 | tri);
	}
}
//:~