/*
 *  SahWorldInterface.cpp
 *  testsah
 *
 *  Created by jian zhang on 5/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "SahWorldInterface.h"
#include <CudaDynamicWorld.h>
#include <SahTetrahedronSystem.h>
#include <tetmesh.h>
#include <SahGlobal.h>

SahWorldInterface::SahWorldInterface() {}
SahWorldInterface::~SahWorldInterface() {}

void SahWorldInterface::create(CudaDynamicWorld * world)
{
#if COLLIDEJUST
    return DynamicWorldInterface::create(world);
#endif
    SahTetrahedronSystem * tetra = new SahTetrahedronSystem;
	if(!readMeshFromFile(tetra)) createTestMesh(tetra);
	// resetVelocity(tetra);
	tetra->setTotalMass(4000.f);
	world->addTetrahedronSystem(tetra);
}

bool SahWorldInterface::readMeshFromFile(SahTetrahedronSystem * mesh)
{ 
	if(BaseFile::InvalidFilename(SahGlobal::FileName)) 
		return false;
		
	if(!BaseFile::FileExists(SahGlobal::FileName)) {
		SahGlobal::FileName = "unknown";
		return false;
	}
	
	ATetrahedronMesh meshData;
	HesperisFile hes;
	hes.setReadComponent(HesperisFile::RTetra);
	hes.addTetrahedron("tetra_0", &meshData);
	if(!hes.open(SahGlobal::FileName)) return false;
	hes.close();
	
	std::cout<<" nt "<<meshData.numTetrahedrons();
	std::cout<<" nv "<<meshData.numPoints();
	
	mesh->generateFromData(&meshData);
	return true;
}

void SahWorldInterface::createTestMesh(SahTetrahedronSystem * mesh)
{ 
	std::cout<<"test mesh num points "<<TetraNumVertices<<"\n";
	std::cout<<"num tetrahedrons "<<TetraNumTetrahedrons<<"\n";
	
	mesh->create(TetraNumTetrahedrons+100, TetraNumVertices+400);
	
	unsigned i;
	Vector3F p;
	for(i=0; i<TetraNumVertices; i++) {
	    p.set(TetraP[i][0], TetraP[i][1], TetraP[i][2]);
	    mesh->addPoint(&p.x);
	}
	
	for(i=0; i<TetraNumTetrahedrons; i++)
	    mesh->addTetrahedron(TetraIndices[i][0], TetraIndices[i][1], TetraIndices[i][2], TetraIndices[i][3]);
	
	mesh->setAnchoredPoint(89, 0);
	mesh->setAnchoredPoint(63, 20);
	mesh->setAnchoredPoint(71, 9);
	mesh->setAnchoredPoint(95, 78);
}