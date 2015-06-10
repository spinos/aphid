#include "AdeniumInterface.h"
#include "AdeniumWorld.h"
#include <HesperisFile.h>
#include <GeometryArray.h>
#include <SahBuilder.h>
#include <BvhTriangleSystem.h>
#include <iostream>
std::string AdeniumInterface::FileName("");
AdeniumInterface::AdeniumInterface() {}
AdeniumInterface::~AdeniumInterface() {}

void AdeniumInterface::create(AdeniumWorld * world)
{
    world->setBvhBuilder(new SahBuilder);
    readTriangleMeshFromFile(world);
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
    
    if(triangleMeshes.numGeometries() < 1) return false;
    std::cout<<" n tri mesh "<<triangleMeshes.numGeometries();
	
    world->addTriangleSystem(new BvhTriangleSystem((ATriangleMesh *)triangleMeshes.geometry(0)));
	return true;
}
