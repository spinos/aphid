#include "FEMWorldInterface.h"
#include <AllMath.h>
#include <CudaDynamicWorld.h>
#include <CudaTetrahedronSystem.h>
#include <tetmesh.h>
FEMWorldInterface::FEMWorldInterface() {}
FEMWorldInterface::~FEMWorldInterface() {}

void FEMWorldInterface::create(CudaDynamicWorld * world)
{
    std::cout<<"num points "<<TetraNumVertices<<"\n";
	std::cout<<"num tetrahedrons "<<TetraNumTetrahedrons<<"\n";
	
    CudaTetrahedronSystem * tetra = new CudaTetrahedronSystem;
	tetra->create(TetraNumTetrahedrons+100, TetraNumVertices+400);
	float * hv = &tetra->hostV()[0];
	float vrx, vry, vrz, vr;
	float vy = 3.95f;
	unsigned i;
	Vector3F p;
	for(i=0; i<TetraNumVertices; i++) {
	    p.set(TetraP[i][0], TetraP[i][1], TetraP[i][2]);
	    tetra->addPoint(&p.x);
	    
	    vrx = 0.725f * (RandomF01() - .5f);
		vry = 1.f  * (RandomF01() + 1.f)  * vy;
		vrz = 0.732f * (RandomF01() - .5f);
		vr = 0.13f * RandomF01();
		
		vy = -vy;
			
	    hv[0] = 0.f;//vrx + vr;
		hv[1] = 0.f;//vry;
		hv[2] = 0.f;//vrz - vr;
		hv+=3;
	}
	
	for(i=0; i<TetraNumTetrahedrons; i++) {
	    tetra->addTetrahedron(TetraIndices[i][0], TetraIndices[i][1], TetraIndices[i][2], TetraIndices[i][3]);
	}
	
	world->addTetrahedronSystem(tetra);
}
