#include "FEMWorldInterface.h"
#include <AllMath.h>
#include <CudaDynamicWorld.h>
#include <FEMTetrahedronSystem.h>
#include <tetmesh.h>
#include "FemGlobal.h"
#include <DynGlobal.h>
#include <SahBuilder.h>
#include <GeometryArray.h>
#include <TriangleSystem.h>

FEMWorldInterface::FEMWorldInterface() {}
FEMWorldInterface::~FEMWorldInterface() {}

void FEMWorldInterface::create(CudaDynamicWorld * world)
{
#if COLLIDEJUST
    return DynamicWorldInterface::create(world);
#endif
    world->setBvhBuilder(new SahBuilder);
	readTetrahedronMeshFromFile(world);
    readTriangleMeshFromFile(world);
}

bool FEMWorldInterface::readTetrahedronMeshFromFile(CudaDynamicWorld * world)
{
	if(BaseFile::InvalidFilename(FemGlobal::FileName)) 
		return false;
		
	if(!BaseFile::FileExists(FemGlobal::FileName)) {
		FemGlobal::FileName = "unknown";
		return false;
	}
	
	ATetrahedronMesh meshData;
	HesperisFile hes;
	hes.setReadComponent(HesperisFile::RTetra);
	hes.addTetrahedron("tetra_0", &meshData);
	if(!hes.open(FemGlobal::FileName)) return false;
	hes.close();
	
	std::cout<<" n tetrahedron: "<<meshData.numTetrahedrons()
	<<"\n n vertex: "<<meshData.numPoints()
	<<"\n";
	
    FEMTetrahedronSystem * mesh = new FEMTetrahedronSystem(&meshData);
    mesh->resetVelocity();
	world->addTetrahedronSystem(mesh);
	
	return true;
}
/*
void FEMWorldInterface::createTestMesh(FEMTetrahedronSystem * mesh)
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
*/
bool FEMWorldInterface::readTriangleMeshFromFile(CudaDynamicWorld * world)
{
    if(BaseFile::InvalidFilename(FemGlobal::FileName)) 
		return false;
		
	if(!BaseFile::FileExists(FemGlobal::FileName)) {
		FemGlobal::FileName = "unknown";
		return false;
	}
    
    HesperisFile hes;
	hes.setReadComponent(HesperisFile::RTri);
	if(!hes.open(FemGlobal::FileName)) return false;
	hes.close();
	
    GeometryArray triangleMeshes;
	hes.extractTriangleMeshes(&triangleMeshes);
    
    if(triangleMeshes.numGeometries() < 1) return false;
    std::cout<<" n tri mesh "<<triangleMeshes.numGeometries();
	
    world->addTriangleSystem(new TriangleSystem((ATriangleMesh *)triangleMeshes.geometry(0)));
	return true;
}
//:~