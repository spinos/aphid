#include "FEMWorldInterface.h"
#include <AllMath.h>
#include <CudaDynamicWorld.h>
#include <FEMTetrahedronSystem.h>
#include <tetmesh.h>
#include "FemGlobal.h"
#include <DynGlobal.h>
#include <SahBuilder.h>

FEMWorldInterface::FEMWorldInterface() {}
FEMWorldInterface::~FEMWorldInterface() {}

void FEMWorldInterface::create(CudaDynamicWorld * world)
{
#if COLLIDEJUST
    return DynamicWorldInterface::create(world);
#endif
    world->setBvhBuilder(new SahBuilder);
	
    FEMTetrahedronSystem * tetra = new FEMTetrahedronSystem;
	if(!readMeshFromFile(tetra)) createTestMesh(tetra);
	resetVelocity(tetra);
	tetra->setTotalMass(12000.f);
	world->addTetrahedronSystem(tetra);
}

bool FEMWorldInterface::readMeshFromFile(FEMTetrahedronSystem * mesh)
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
	
	std::cout<<" nt "<<meshData.numTetrahedrons();
	std::cout<<" nv "<<meshData.numPoints();
	
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
//:~