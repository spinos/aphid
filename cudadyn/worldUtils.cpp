#include "worldUtils.h"
#include <AllMath.h>
#include "CudaDynamicWorld.h"
#include <CudaTetrahedronSystem.h>
#include <GeoDrawer.h>

#define GRDW 55
#define GRDH 55
#define NTET 3600

void createWorld(CudaDynamicWorld * world)
{
    CudaTetrahedronSystem * tetra = new CudaTetrahedronSystem;
	tetra->create(NTET, 1.f, 1.f);
	float * hv = &tetra->hostV()[0];
	
	unsigned i, j;
	float vy = 3.95f;
	float vrx, vry, vrz, vr, vs;
	for(j=0; j < GRDH; j++) {
		for(i=0; i<GRDW; i++) {
		    vs = 1.75f + RandomF01() * 1.5f;
			Vector3F base(9.3f * i, 9.3f * j, 0.f * j);
			Vector3F right = base + Vector3F(1.75f, 0.f, 0.f) * vs;
			Vector3F front = base + Vector3F(0.f, 0.f, 1.75f) * vs;
			Vector3F top = base + Vector3F(0.f, 1.75f, 0.f) * vs;
			if((j&1)==0) {
			    right.y = top.y-.1f;
			}
			else {
			    base.x -= .085f * vs;
			}
			
			vrx = 0.725f * (RandomF01() - .5f);
			vry = 1.f  * (RandomF01() + 1.f)  * vy;
			vrz = 0.732f * (RandomF01() - .5f);
			vr = 0.13f * RandomF01();
			
			tetra->addPoint(&base.x);
			hv[0] = vrx + vr;
			hv[1] = vry;
			hv[2] = vrz - vr;
			hv+=3;
			tetra->addPoint(&right.x);
			hv[0] = vrx - vr;
			hv[1] = vry;
			hv[2] = vrz + vr;
			hv+=3;
			tetra->addPoint(&top.x);
			hv[0] = vrx + vr;
			hv[1] = vry;
			hv[2] = vrz + vr;
			hv+=3;
			tetra->addPoint(&front.x);
			hv[0] = vrx - vr;
			hv[1] = vry;
			hv[2] = vrz - vr;
			hv+=3;

			unsigned b = (j * GRDW + i) * 4;
			tetra->addTetrahedron(b, b+1, b+2, b+3);
			
			tetra->addTriangle(b, b+2, b+1);
			tetra->addTriangle(b, b+1, b+3);
			tetra->addTriangle(b, b+3, b+2);
			tetra->addTriangle(b+1, b+2, b+3);
//	 2
//	 | \ 
//	 |  \
//	 0 - 1
//  /
// 3 		
		}
		vy = -vy;
	}
	
	world->addTetrahedronSystem(tetra);
}

void drawTetra(TetrahedronSystem * tetra)
{
	glColor3f(0.3f, 0.4f, 0.3f);
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)tetra->hostX());
	glDrawElements(GL_TRIANGLES, tetra->numTriangleFaceVertices(), GL_UNSIGNED_INT, tetra->hostTriangleIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	glColor3f(0.1f, 0.4f, 0.f);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)tetra->hostX());
	glDrawElements(GL_TRIANGLES, tetra->numTriangleFaceVertices(), GL_UNSIGNED_INT, tetra->hostTriangleIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
}

void drawWorld(CudaDynamicWorld * world)
{
    const unsigned nobj = world->numObjects();
    if(nobj<1) return;
    
    unsigned i;
    for(i=0; i< nobj; i++) {
        CudaTetrahedronSystem * tetra = world->tetradedron(i);
        tetra->sendXToHost();
        drawTetra(tetra);
    }
	
	world->dbgDraw();
}
