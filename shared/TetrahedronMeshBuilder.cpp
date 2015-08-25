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