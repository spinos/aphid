#include <QtGui>

#include <gl_heads.h>
#include "glwidget.h"
#include "simpleMesh.h"
#include <KdTreeDrawer.h>
#include <CUDABuffer.h>
#include <BvhSolver.h>
#include "bvh_common.h"
#include <radixsort_implement.h>
#include <CudaBase.h>
#include "CudaLinearBvh.h"
#include "CudaParticleSystem.h"
#include "CudaTetrahedronSystem.h"
#include "CudaBroadphase.h"
#include "DrawBvh.h"
#include "rayTest.h"
#include "bvh_dbg.h"

#define IRAYDIM 33

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_mesh = new SimpleMesh;
	m_ray = new RayTest;
	m_particles = new CudaParticleSystem;
	m_particles->createParticles(10 * 20 * 10);

	int i, j, k;
	Vector3F * p = (Vector3F *)m_particles->position();
	for(k=0; k < 10; k++)
	    for(j=0; j < 20; j++)
	        for(i=0; i < 10; i++) {
	            p->x = 2.5f * i;
	            p->y = 2.5f * j + 100.f;
	            p->z = 2.5f * k;
	            p++;
	        }
	Vector3F * v = (Vector3F *)m_particles->velocity();
	for(i=0; i < 2000; i++) v[i].set(0.f, 10.f, 10.f);
	Vector3F * f = (Vector3F *)m_particles->force();
	for(i=0; i < 2000; i++) f[i].setZero();
	
	m_solver = new BvhSolver;
	m_displayLevel = 3;

	qDebug()<<"num vertices "<<m_mesh->numVertices();
	qDebug()<<"num triangles "<<m_mesh->numTriangles();
	qDebug()<<"num ray tests "<<(IRAYDIM * IRAYDIM);
	
	m_tetra = new CudaTetrahedronSystem;
	m_tetra->create(2200, 1.f, 1.f);
	
	float * hv = &m_tetra->hostV()[0];
	
	const unsigned grdx = 43;
	for(j=0; j < 32; j++) {
		for(i=0; i<grdx; i++) {
			Vector3F base(2.f * i, 2.f * j, 0.5f * i);
			Vector3F right = base + Vector3F(1.75f, 0.f, 0.f);
			Vector3F front = base + Vector3F(0.f, 0.f, 1.75f);
			Vector3F top = base + Vector3F(0.f, 1.75f, 0.f);
			
			m_tetra->addPoint(&base.x);
			hv[0] = 30.f * (((float)(rand() % 199))/199.f - .5f);
			hv[1] = 30.f * (((float)(rand() % 199))/199.f - .5f);
			hv[2] = 33.f * (((float)(rand() % 199))/199.f - .5f);
			hv+=3;
			m_tetra->addPoint(&right.x);
			hv[0] = 30.f * (((float)(rand() % 199))/199.f - .5f);
			hv[1] = 38.f * (((float)(rand() % 199))/199.f - .5f);
			hv[2] = 35.f * (((float)(rand() % 199))/199.f - .5f);
			hv+=3;
			m_tetra->addPoint(&top.x);
			hv[0] = 30.f * (((float)(rand() % 199))/199.f - .5f);
			hv[1] = 38.f * (((float)(rand() % 199))/199.f - .5f);
			hv[2] = 35.f * (((float)(rand() % 199))/199.f - .5f);
			hv+=3;
			m_tetra->addPoint(&front.x);
			hv[0] = 30.f * (((float)(rand() % 199))/199.f - .5f);
			hv[1] = 38.f * (((float)(rand() % 199))/199.f - .5f);
			hv[2] = 35.f * (((float)(rand() % 199))/199.f - .5f);
			hv+=3;

			unsigned b = (j * grdx + i) * 4;
			m_tetra->addTetrahedron(b, b+1, b+2, b+3);
			
			m_tetra->addTriangle(b, b+2, b+1);
			m_tetra->addTriangle(b, b+1, b+3);
			m_tetra->addTriangle(b, b+3, b+2);
			m_tetra->addTriangle(b+1, b+2, b+3);
//	 2
//	 | \ 
//	 |  \
//	 0 - 1
//  /
// 3 		
		}
	}	
	
	/*for(j=0; j < 32; j++) {
		for(i=0; i<32; i++) {
		    float * v = &m_tetra->hostV()[(j*32 + i)*3];
		    std::cout<<" "<<v[0]<<" "<<v[1]<<" "<<v[2]<<"\n";
		}	
	}*/
	qDebug()<<"tetra n tet "<<m_tetra->numTetradedrons();
	qDebug()<<"tetra n p "<<m_tetra->numPoints();
	qDebug()<<"tetra n tri "<<m_tetra->numTriangles();
	
	m_drawBvh = new DrawBvh;
	m_drawBvh->setDrawer(getDrawer());
	m_drawBvh->setBvh(m_tetra);
	
	m_broadphase = new CudaBroadphase;
	m_broadphase->addBvh(m_tetra);
	
	m_drawBvh->setBroadphase(m_broadphase);
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
	CudaBase::SetDevice();
	m_mesh->initOnDevice();
	m_particles->initOnDevice();
	m_solver->setMesh(m_mesh);
	m_ray->createRays(IRAYDIM, IRAYDIM);
	m_solver->setRay(m_ray);
	m_solver->setParticleSystem(m_particles);
	
#ifdef BVHSOLVER_DBG_DRAW	
	CudaLinearBvh * bvh = m_mesh->bvh();
	qDebug()<<" bvh used memory "<<bvh->usedMemory()<<" bytes";
	m_displayLeafAabbs = new BaseBuffer;
	m_displayLeafAabbs->create(bvh->numLeafNodes() * sizeof(Aabb));
	m_displayInternalAabbs = new BaseBuffer;
	m_displayInternalAabbs->create(bvh->numInternalNodes() * sizeof(Aabb));
	m_displayLeafHash = new BaseBuffer;
	m_displayLeafHash->create(bvh->numLeafNodes() * sizeof(KeyValuePair));
	m_displayInternalDistance = new BaseBuffer;
	m_displayInternalDistance->create(bvh->numInternalNodes() * sizeof(int));
	m_internalChildIndices = new BaseBuffer;
	m_internalChildIndices->create(bvh->numInternalNodes() * sizeof(int2));
	m_rootNodeInd = new int[1];
	m_solver->setHostPtrs(m_displayLeafAabbs,
                m_displayInternalAabbs,
                m_displayInternalDistance,
                m_displayLeafHash,
                m_internalChildIndices,
                m_rootNodeInd);
#endif

    m_tetra->initOnDevice();
	m_broadphase->initOnDevice();
	
	m_broadphase->computeOverlappingPairs();
	m_drawBvh->printPairCounts();
	
	// connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	// connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	// drawMesh();
	drawTetra();
	const float t = (float)elapsedTime();
	m_mesh->setAlpha(t/290.f);
	m_ray->setAlpha(t/230.f);
	// qDebug()<<"drawn in "<<deltaTime();
	// internalTimer()->start();
}

void GLWidget::drawMesh()
{
	CudaLinearBvh * bvh = m_mesh->bvh();
	Aabb ab = bvh->bound();
	if(ab.low.x < -1e8 || ab.low.x > 1e8) std::cout<<" invalid big box "<<aabb_str(ab);
	
#ifdef BVHSOLVER_DBG_DRAW
    unsigned numInternal = bvh->numInternalNodes();
	debugDraw(*m_rootNodeInd, numInternal);
#else
	GeoDrawer * dr = getDrawer();
    BoundingBox bb; 
	bb.setMin(ab.low.x, ab.low.y, ab.low.z);
	bb.setMax(ab.high.x, ab.high.y, ab.high.z);
	
    dr->boundingBox(bb);
#endif

#ifdef BVHSOLVER_DRAW_MESH
    glColor3f(0.f, 0.3f, 0.5f);
	//internalTimer()->stop();
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_mesh->vertices());
	glDrawElements(GL_TRIANGLES, m_mesh->numTriangleFaceVertices(), GL_UNSIGNED_INT, m_mesh->triangleIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
#endif
}

void GLWidget::drawParticles()
{
	glColor3f(0.f, 0.9f, 0.3f);
    
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_particles->position());
    glDrawArrays(GL_POINTS, 0, m_particles->numParticles());

	glDisableClientState(GL_VERTEX_ARRAY);
}

inline int isLeafNode(int index) 
{ return (index >> 31 == 0); }

inline int getIndexWithInternalNodeMarkerRemoved(int index) 
{ return index & (~0x80000000); }


void GLWidget::debugDraw(unsigned rootInd, unsigned numInternal)
{
#ifdef BVHSOLVER_DBG_DRAW
	if(!m_solver->isValid()) return;
	
	GeoDrawer * dr = getDrawer();
	// qDebug()<<" root at "<< (*m_rootNodeInd & (~0x80000000));
	Aabb ab; BoundingBox bb; 
#ifdef BVHSOLVER_DBG_DRAW_BOX
	Aabb * internalBoxes = (Aabb *)m_displayInternalAabbs->data();
	Aabb * leafBoxes = (Aabb *)m_displayLeafAabbs->data();
	int * levels = (int *)m_displayInternalDistance->data();
	KeyValuePair * leafHash = (KeyValuePair *)m_displayLeafHash->data();
	int2 * internalNodeChildIndices = (int2 *)m_internalChildIndices->data();
	
	// std::cout<<" "<<numInternal/2<<" "<<byte_to_binary(leafHash[numInternal/2].key)<<" "<<leafHash[numInternal/2].value<<"\n";
	
	int stack[128];
	stack[0] = rootInd;
	int stackSize = 1;
	int maxStack = 1;
	int touchedLeaf = 0;
	int touchedInternal = 0;
	while(stackSize > 0) {
		int internalOrLeafNodeIndex = stack[ stackSize - 1 ];
		stackSize--;
		
		int isLeaf = isLeafNode(internalOrLeafNodeIndex);	//Internal node if false
		uint bvhNodeIndex = getIndexWithInternalNodeMarkerRemoved(internalOrLeafNodeIndex);
		
		int bvhRigidIndex = (isLeaf) ? leafHash[bvhNodeIndex].value : -1;
		
		Aabb bvhNodeAabb = (isLeaf) ? leafBoxes[bvhRigidIndex] : internalBoxes[bvhNodeIndex];

		{
			if(isLeaf) {
#ifdef BVHSOLVER_DBG_DRAW_LEAFBOX
				glColor3f(.5, 0., 0.);
				ab = bvhNodeAabb;
				bb.setMin(ab.low.x, ab.low.y, ab.low.z);
				bb.setMax(ab.high.x, ab.high.y, ab.high.z);
				dr->boundingBox(bb);
#endif
				touchedLeaf++;
			}
			else {
#ifdef BVHSOLVER_DBG_DRAW_INTERNALBOX_TO_LEVEL
				glColor3f(.5, .65, 0.);
				if(levels[bvhNodeIndex] > m_displayLevel) continue;
				ab = bvhNodeAabb;
				bb.setMin(ab.low.x, ab.low.y, ab.low.z);
				bb.setMax(ab.high.x, ab.high.y, ab.high.z);
				dr->boundingBox(bb);
#endif
				touchedInternal++;
				if(stackSize + 2 > 128)
				{
					//Error
				}
				else
				{
				    stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].x;
					stackSize++;
					stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].y;
					stackSize++;
					
					if(stackSize > maxStack) maxStack = stackSize;
				}
			}
		}
		
	} 
	
	//qDebug()<<"max stack "<<maxStack<<" touch leaf "<<touchedLeaf<<" touchedInternal "<<touchedInternal;
#endif	

#ifdef BVHSOLVER_DBG_DRAW_LEAFHASH
	glBegin(GL_LINES);
	int nzero = 0;
	for(unsigned i=0; i < numInternal; i++) {
		float red = (float)i/(float)numInternal;
		
		if(leafHash[i].value > numInternal) {
			qDebug()<<"invalid hash value "<<leafHash[i].value;
			nzero++;
			continue;
		}
		
		glColor3f(red, 1.f - red, 0.f);
		Aabb a0 = leafBoxes[leafHash[i].value];
		glVertex3f(a0.low.x * 0.5f + a0.high.x * 0.5f, a0.low.y * 0.5f + a0.high.y * 0.5f + 0.2f, a0.low.z * 0.5f + a0.high.z * 0.5f);
        
		Aabb a1 = leafBoxes[leafHash[i+1].value];
		glVertex3f(a1.low.x * 0.5f + a1.high.x * 0.5f, a1.low.y * 0.5f + a1.high.y * 0.5f + 0.2f, a1.low.z * 0.5f + a1.high.z * 0.5f);
        
	}
	glEnd();	
	if(nzero > 0) qDebug()<<"n zero code "<<nzero;
#endif

#endif

#ifdef BVHSOLVER_DBG_DRAW_RAY
    RayInfo * rays = m_ray->getRays();
	const unsigned nr = m_ray->numRays();
	glColor3f(0.f, 0.5f, 0.2f);
	glBegin(GL_LINES);
	for(unsigned i=0; i < nr; i++) {
		RayInfo & r = rays[i];
		glVertex3f(r.origin.x, r.origin.y, r.origin.z);
		glVertex3f(r.destiny.x, r.destiny.y, r.destiny.z);
		
		Vector3F a(r.destiny.x - r.origin.x, r.destiny.y - r.origin.y, r.destiny.z - r.origin.z);
		// qDebug()<<" "<<a.length();
	}
	glEnd();
#endif
}

void GLWidget::drawTetra()
{
	m_broadphase->computeOverlappingPairs();
	
	glColor3f(0.1f, 0.4f, 0.3f);
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_tetra->hostX());
	glDrawElements(GL_TRIANGLES, m_tetra->numTriangleFaceVertices(), GL_UNSIGNED_INT, m_tetra->hostTriangleIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
	
	glColor3f(0.2f, 0.2f, 0.3f);
    
	m_drawBvh->bound();
	m_drawBvh->leaf();
	// m_drawBvh->hash();
	// m_drawBvh->hierarch();
	m_drawBvh->pairs();
}

void GLWidget::clientSelect(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::clientDeselect(QMouseEvent */*event*/) 
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::clientMouseInput(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
	switch (event->key()) {
		case Qt::Key_A:
			m_displayLevel++;
			m_drawBvh->addDispalyLevel();
			break;
		case Qt::Key_D:
			m_displayLevel--;
			m_drawBvh->minusDispalyLevel();
			break;
		case Qt::Key_W:
			internalTimer()->stop();
			break;
		case Qt::Key_S:
			internalTimer()->start();
			break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}
