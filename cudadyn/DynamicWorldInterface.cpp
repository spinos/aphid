#include "DynamicWorldInterface.h"
#include "CudaDynamicWorld.h"
#include <CudaTetrahedronSystem.h>
#include <CudaBroadphase.h>
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include <AllMath.h>
#include <GeoDrawer.h>
#include <stripedModel.h>

#define GRDW 57
#define GRDH 57
#define NTET 3600
#define NPNT 14400

struct A {
    // mat33 Ke[4][4];
    mat33 Re;
    //float3 B[4]; 
    //float3 e1, e2, e3;
    //float volume;
    //float plastic[6];
};

DynamicWorldInterface::DynamicWorldInterface() 
{
    std::cout<<" size of A "<<sizeof(A)<<"\n";
    m_boxes = new BaseBuffer;
    m_pairCache = new BaseBuffer;
}

DynamicWorldInterface::~DynamicWorldInterface() {}

void DynamicWorldInterface::create(CudaDynamicWorld * world)
{
    CudaTetrahedronSystem * tetra = new CudaTetrahedronSystem;
	tetra->create(NTET, NPNT);
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
		}
		vy = -vy;
	}
	
	world->addTetrahedronSystem(tetra);
}

void DynamicWorldInterface::draw(TetrahedronSystem * tetra)
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

void DynamicWorldInterface::draw(CudaDynamicWorld * world)
{
    const unsigned nobj = world->numObjects();
    if(nobj<1) return;
    
    unsigned i;
    for(i=0; i< nobj; i++) {
        CudaTetrahedronSystem * tetra = world->tetradedron(i);
        tetra->sendXToHost();
        draw(tetra);
    }
}

void DynamicWorldInterface::draw(CudaDynamicWorld * world, GeoDrawer * drawer)
{
    glDisable(GL_DEPTH_TEST);
    draw(world);
    showOverlappingPairs(world, drawer);
}

void DynamicWorldInterface::showOverlappingPairs(CudaDynamicWorld * world, GeoDrawer * drawer)
{
    CudaBroadphase * broadphase = world->broadphase();
    const unsigned cacheLength = broadphase->pairCacheLength();
	if(cacheLength < 1) return;
	
	const unsigned nb = broadphase->numBoxes();
	m_boxes->create(nb * 24);
	
	broadphase->getBoxes(m_boxes);
	
	Aabb * boxes = (Aabb *)m_boxes->data();
	Aabb abox;
	BoundingBox ab, bb;
	unsigned i;
	drawer->setColor(0.f, 0.1f, 0.3f);
	
	m_pairCache->create(broadphase->numUniquePairs() * 8);
	CUDABuffer * uniquePairs = broadphase->overlappingPairBuf();
	uniquePairs->deviceToHost(m_pairCache->data(), m_pairCache->bufferSize());
	unsigned * pc = (unsigned *)m_pairCache->data();
	
	unsigned objectI;
	for(i=0; i < broadphase->numUniquePairs(); i++) {
	    objectI = extractObjectInd(pc[i * 2]);
	    abox = boxes[broadphase->objectStart(objectI) + extractElementInd(pc[i * 2])];
	    
		bb.setMin(abox.low.x, abox.low.y, abox.low.z);
		bb.setMax(abox.high.x, abox.high.y, abox.high.z);
	    
	    objectI = extractObjectInd(pc[i * 2 + 1]);
	    abox = boxes[broadphase->objectStart(objectI) + extractElementInd(pc[i * 2 + 1])];
	    
	    ab.setMin(abox.low.x, abox.low.y, abox.low.z);
		ab.setMax(abox.high.x, abox.high.y, abox.high.z);
		
		drawer->arrow(bb.center(), ab.center());
		
		bb.expandBy(ab);
		
		// m_drawer->boundingBox(bb);
	}
}

