#include "FEMWorldInterface.h"
#include <AllMath.h>
#include <CudaDynamicWorld.h>
#include <FEMTetrahedronSystem.h>
#include <tetmesh.h>
#include "FemGlobal.h"
#include <DynGlobal.h>
#include <SahBuilder.h>
#include <GeometryArray.h>
#include <BvhTriangleSystem.h>
#include <IVelocityFile.h>
#include <ATetrahedronMeshGroup.h>

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
	
	HesperisFile hes;
	hes.setReadComponent(HesperisFile::RTetra);
	if(!hes.open(FemGlobal::FileName)) return false;
	hes.close();
	
    GeometryArray tetrahedronGeos;
	hes.extractTetrahedronMeshes(&tetrahedronGeos);
    
    unsigned n = tetrahedronGeos.numGeometries();
    if(n<1) std::cout<<"\n ERROR: found zero tetrahedron mesh ";
    ATetrahedronMeshGroup * meshData = (ATetrahedronMeshGroup *)tetrahedronGeos.geometry(0);
    std::cout<<"\n mesh["<<0<<"]"
	    <<"\n "<<meshData->verbosestr()
        <<"\n";
        
    FEMTetrahedronSystem * mesh = new FEMTetrahedronSystem(meshData);
    mesh->resetVelocity();
    world->addTetrahedronSystem(mesh);
    
	StripeMap stripemap;
    
    stripemap.create(meshData);
    stripemap.setLastIndex(meshData->numIndices());
    stripemap.computeTetrahedronInStripe(mesh->hostElementValue(),
                                     mesh->numTetrahedrons());
									 
	delete meshData;
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
	
    world->addTriangleSystem(new BvhTriangleSystem((ATriangleMesh *)triangleMeshes.geometry(0)));
	return true;
}

bool FEMWorldInterface::useVelocityFile(CudaDynamicWorld * world)
{
    IVelocityFile * velfile = new IVelocityFile;
    if(!velfile->open("./velocity.tmp")) return false;
    
    const unsigned nv = velfile->readNumPoints();
    std::cout<<"\n cached n velocity "<<nv;
    if(nv != world->totalNumPoints()) {
        std::cout<<" not match world n point "<<world->totalNumPoints();
        return false;
    }
    
    if(!velfile->readFrameRange()) return false;
    
	velfile->readMaxSpeed();
	world->updateSpeedLimit(velfile->maxSpeed());
    velfile->frameBegin();
    CudaDynamicWorld::VelocityCache = velfile;
    world->beginCache();
    velfile->beginCountNumFramesPlayed();
    return true;
}

void FEMWorldInterface::updateDensity(float x)
{
	TetrahedronSystem::Density = x;
    FEMTetrahedronSystem::SetNeedMass();
}

bool FEMWorldInterface::HasVelocityFile()
{ return CudaDynamicWorld::VelocityCache != 0; }

int FEMWorldInterface::VelocityFileBegin()
{ return CudaDynamicWorld::VelocityCache->FirstFrame; }

int FEMWorldInterface::VelocityFileEnd()
{ return CudaDynamicWorld::VelocityCache->LastFrame; }

int FEMWorldInterface::CurrentFrame()
{ return CudaDynamicWorld::VelocityCache->currentFrame(); }

void FEMWorldInterface::updateStiffnessMapEnds(float a, float b)
{
    FEMTetrahedronSystem::SplineMap.setStart(a);
    FEMTetrahedronSystem::SplineMap.setEnd(b);
	FEMTetrahedronSystem::SetNeedElasticity();
}

void FEMWorldInterface::updateStiffnessMapLeft(float x, float y)
{ 
	FEMTetrahedronSystem::SplineMap.setLeftControl(x, y); 
	FEMTetrahedronSystem::SetNeedElasticity();
}

void FEMWorldInterface::updateStiffnessMapRight(float x, float y)
{ 
	FEMTetrahedronSystem::SplineMap.setRightControl(x, y); 
	FEMTetrahedronSystem::SetNeedElasticity();
}

void FEMWorldInterface::updateYoungsModulus(float x)
{ 
	FEMTetrahedronSystem::YoungsModulus = x;
	FEMTetrahedronSystem::SetNeedElasticity();
}
//:~