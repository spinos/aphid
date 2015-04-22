#include "FEMWorldInterface.h"
#include <AllMath.h>
#include <CudaDynamicWorld.h>
#include <FEMTetrahedronSystem.h>
#include <tetmesh.h>
#include "FemGlobal.h"
#define COLLIDEJUST 0

FEMWorldInterface::FEMWorldInterface() {}
FEMWorldInterface::~FEMWorldInterface() {}

void FEMWorldInterface::create(CudaDynamicWorld * world)
{
#if COLLIDEJUST
    return DynamicWorldInterface::create(world);
#endif
    FEMTetrahedronSystem * tetra = new FEMTetrahedronSystem;
	createTestMesh(tetra);
	world->addTetrahedronSystem(tetra);
}

void FEMWorldInterface::createTestMesh(FEMTetrahedronSystem * mesh)
{
	std::cout<<"test mesh num points "<<TetraNumVertices<<"\n";
	std::cout<<"num tetrahedrons "<<TetraNumTetrahedrons<<"\n";
	
	mesh->create(TetraNumTetrahedrons+100, TetraNumVertices+400);
	float * hv = &mesh->hostV()[0];
	float vrx, vry, vrz, vr;
	float vy = 1.695f;
	unsigned i;
	Vector3F p, q;
	for(i=0; i<TetraNumVertices; i++) {
	    p.set(TetraP[i][0], TetraP[i][1], TetraP[i][2]);
	    mesh->addPoint(&p.x);
	    
	    vrx = 0.0932f * (RandomF01() - .5f);
		vry = .5f * (RandomF01() + 1.f)  * vy;
		vrz = 0.0932f * (RandomF01() - .5f);
		vr = 0.013f * RandomF01();
		
		vy = -vy;
			
	    hv[0] = 0.f;//vrx + vr;
		hv[1] = 0.f;
		hv[2] = 0.f;//vrz - vr;
		hv+=3;
	}
	
	for(i=0; i<TetraNumTetrahedrons; i++)
	    mesh->addTetrahedron(TetraIndices[i][0], TetraIndices[i][1], TetraIndices[i][2], TetraIndices[i][3]);
	
	mesh->setAnchoredPoint(122, 0);
	mesh->setAnchoredPoint(123, 20);
	mesh->setAnchoredPoint(116, 9);
	mesh->setAnchoredPoint(124, 78);
	mesh->setTotalMass(1000.f);
}
//:~