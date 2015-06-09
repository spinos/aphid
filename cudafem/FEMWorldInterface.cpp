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
	
    FEMTetrahedronSystem * tetra = new FEMTetrahedronSystem;
	if(!readTetrahedronMeshFromFile(tetra)) createTestMesh(tetra);
	resetVelocity(tetra);
	world->addTetrahedronSystem(tetra);
    
    readTriangleMeshFromFile(world);
}

bool FEMWorldInterface::readTetrahedronMeshFromFile(FEMTetrahedronSystem * mesh)
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
	
	mesh->generateFromData(&meshData);
	return true;
}

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

void FEMWorldInterface::resetVelocity(FEMTetrahedronSystem * mesh)
{
	Vector3F * hv = (Vector3F *)mesh->hostV();
	unsigned i;
	const unsigned n = mesh->numPoints();
	for(i=0; i< n; i++) hv[i].setZero();
}

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