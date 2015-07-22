#include "AdeniumInterface.h"
#include "AdeniumWorld.h"
#include <HesperisFile.h>
#include <GeometryArray.h>
#include <SahBuilder.h>
#include <BvhTriangleSystem.h>
#include <WorldDbgDraw.h>
#include <iostream>
std::string AdeniumInterface::FileName("");
H5FileIn AdeniumInterface::BakeFile;
AdeniumInterface::AdeniumInterface() {}
AdeniumInterface::~AdeniumInterface() {}

void AdeniumInterface::create(AdeniumWorld * world)
{
    world->setBvhBuilder(new SahBuilder);
    readTriangleMeshFromFile(world);
    readTetrahedronMeshFromFile(world);
}

bool AdeniumInterface::readTriangleMeshFromFile(AdeniumWorld * world)
{
    if(BaseFile::InvalidFilename(FileName)) 
		return false;
		
	if(!BaseFile::FileExists(FileName)) {
		FileName = "unknown";
		return false;
	}
    
    HesperisFile hes;
	hes.setReadComponent(HesperisFile::RTri);
	if(!hes.open(FileName)) return false;
	hes.close();
	
    GeometryArray triangleMeshes;
	hes.extractTriangleMeshes(&triangleMeshes);
    
    if(triangleMeshes.numGeometries() < 1) {
        std::cout<<" no triangle mesh discovered\n";
        return false;
    }
    std::cout<<" n tri mesh "<<triangleMeshes.numGeometries();
	
    ATriangleMesh * gm = (ATriangleMesh *)triangleMeshes.geometry(0);
    world->setRestMesh(gm);
    world->addTriangleSystem(new BvhTriangleSystem(gm));
    return true;
}

bool AdeniumInterface::readTetrahedronMeshFromFile(AdeniumWorld * world)
{
    if(BaseFile::InvalidFilename(FileName)) 
		return false;
		
	if(!BaseFile::FileExists(FileName)) {
		FileName = "unknown";
		return false;
	}
    
    HesperisFile hes;
	hes.setReadComponent(HesperisFile::RTetra);
	if(!hes.open(FileName)) return false;
	hes.close();
    
    GeometryArray tetraMeshes;
	hes.extractTetrahedronMeshes(&tetraMeshes);
    
    if(tetraMeshes.numGeometries() < 1) {
        std::cout<<" no tetrahedron mesh discovered\n";
        return false;
    }
    std::cout<<" n tetra mesh "<<tetraMeshes.numGeometries();
	
    world->addTetrahedronMesh((ATetrahedronMesh *)tetraMeshes.geometry(0));
    return true;
}

void AdeniumInterface::changeMaxDisplayLevel(AdeniumWorld * world, int x)
{
	int level = WorldDbgDraw::MaxDrawBvhHierarchyLevel + x;
	if(level<1) level = 1;
	if(level>30) level = 30;
	WorldDbgDraw::MaxDrawBvhHierarchyLevel = level;
}

bool AdeniumInterface::LoadBake(AdeniumWorld * world, const std::string & name)
{
    if(BakeFile.isOpened()) BakeFile.close();
    if(!BakeFile.open(name)) return false;
    
    std::string meshName;
    ATriangleMesh * tri = BakeFile.findBakedMesh(meshName);
    if(!tri) return false;
    
    std::cout<<"mesh name is "<<meshName;
    
    if(!world->matchRestMesh(tri)) {
        std::cout<<" bake failed to match rest mesh!\n";
        return false;
    }
    
    std::cout<<" bake range ("<<BakeFile.FirstFrame
    <<", "<<BakeFile.LastFrame
    <<")\n";
    
    BakeFile.frameBegin();
    world->setDifferenceObject(tri);
    world->beginRecordVelocity(&BakeFile);
    return true;
}

bool AdeniumInterface::ReadBakeFrame(AdeniumWorld * world)
{
    ATriangleMesh * m = world->deformedMesh();
    if(!BakeFile.readFrame(m->points(), m->numPoints()))
        return false;
    world->processFrame(BakeFile.isFrameBegin());
    if(BakeFile.isFrameEnd()) BakeFile.frameBegin();
    else BakeFile.nextFrame();
    return true;
}
//:~
