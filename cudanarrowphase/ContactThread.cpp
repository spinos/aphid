#include <QtCore>
#include "ContactThread.h"
#include <AllMath.h>
#include <CudaTetrahedronSystem.h>
#include <CudaNarrowphase.h>
#include "SimpleContactSolver.h"
#include <CUDABuffer.h>

#define GRDX 99
#define NTET 2500

ContactThread::ContactThread(QObject *parent)
    : BaseSolverThread(parent)
{
    m_tetra = new CudaTetrahedronSystem;
	m_tetra->create(NTET, 1.f, 1.f);
	float * hv = &m_tetra->hostV()[0];
	
	unsigned i, j;
	float vy = .75f;
	float vrx, vry, vrz, vr, vs;
	for(j=0; j < 2; j++) {
		for(i=0; i<GRDX; i++) {
		    vs = 1.5f + (((float)(rand() % 199))/199.f) * 1.5f;
			Vector3F base(12.3f * i, 6.f * j + 1.f, 0.5f * i);
			Vector3F right = base + Vector3F(1.75f, 0.f, 0.7f) * vs;
			Vector3F front = base + Vector3F(0.f, 0.f, 1.75f) * vs;
			Vector3F top = base + Vector3F(0.f, 1.75f, 0.7f) * vs;
			if(j%2==0) top.x += 1.75f;
			
			vrx = 0.725f * (((float)(rand() % 199))/199.f - .5f);
			vry = 1.f  * (((float)(rand() % 199))/199.f + 1.f)  * vy;
			vrz = 0.822f * (((float)(rand() % 199))/199.f - .5f);
			vr = 0.f * (((float)(rand() % 199))/199.f);
			
			m_tetra->addPoint(&base.x);
			hv[0] = vrx + vr;
			hv[1] = vry;
			hv[2] = vrz - vr;
			hv+=3;
			m_tetra->addPoint(&right.x);
			hv[0] = vrx - vr;
			hv[1] = vry;
			hv[2] = vrz + vr;
			hv+=3;
			m_tetra->addPoint(&top.x);
			hv[0] = vrx + vr;
			hv[1] = vry;
			hv[2] = vrz + vr;
			hv+=3;
			m_tetra->addPoint(&front.x);
			hv[0] = vrx - vr;
			hv[1] = vry;
			hv[2] = vrz - vr;
			hv+=3;

			unsigned b = (j * GRDX + i) * 4;
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
		vy = -vy;
	}
	
	qDebug()<<"tetra n tet "<<m_tetra->numTetradedrons();
	qDebug()<<"tetra n p "<<m_tetra->numPoints();
	qDebug()<<"tetra n tri "<<m_tetra->numTriangles();
	
	m_narrowphase = new CudaNarrowphase;
	m_narrowphase->addTetrahedronSystem(m_tetra);
	m_contactSolver = new SimpleContactSolver;
	
	m_hostPairs = new BaseBuffer;
	m_devicePairs = new CUDABuffer;
	
	m_hostPairs->create(GRDX * 8);
	unsigned * pab = (unsigned *)m_hostPairs->data();
	for(i=0; i<GRDX; i++) {
		pab[i*2] = i;
		pab[i*2 + 1] = i + GRDX;
	}
	
}

ContactThread::~ContactThread() {}

void ContactThread::initOnDevice()
{
    m_narrowphase->initOnDevice();
	
	m_devicePairs->create(GRDX * 8);
	m_devicePairs->hostToDevice(m_hostPairs->data(), GRDX *8);
	
	// m_narrowphase->computeContacts(m_devicePairs, GRDX);
	m_contactSolver->initOnDevice();
}

void ContactThread::stepPhysics(float dt)
{
	m_narrowphase->computeContacts(m_devicePairs, GRDX);
	
	m_contactSolver->solveContacts(m_narrowphase->numContacts(),
									m_narrowphase->contactBuffer(),
									m_narrowphase->contactPairsBuffer(),
									m_narrowphase->objectBuffer());
	m_tetra->integrate(0.016667f);
}

CudaTetrahedronSystem * ContactThread::tetra()
{ return m_tetra; }

CudaNarrowphase * ContactThread::narrowphase()
{ return m_narrowphase; }

SimpleContactSolver * ContactThread::contactSolver()
{ return m_contactSolver; }

BaseBuffer * ContactThread::hostPairs()
{ return m_hostPairs; }
//:~