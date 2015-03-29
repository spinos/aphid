#include <QtCore>
#include "ContactThread.h"
#include <AllMath.h>
#include <CudaTetrahedronSystem.h>
#include <CudaNarrowphase.h>
#include "SimpleContactSolver.h"
#include <CUDABuffer.h>

#define GRDW 40
#define GRDH 40
#define NTET 2500

ContactThread::ContactThread(QObject *parent)
    : BaseSolverThread(parent)
{
    m_tetra = new CudaTetrahedronSystem;
	m_tetra->create(NTET, 1.f, 1.f);
	float * hv = &m_tetra->hostV()[0];
	
	unsigned i, j;
	float vy = 2.95f;
	float vrx, vry, vrz, vr, vs;
	for(j=0; j < GRDH; j++) {
		for(i=0; i<GRDW; i++) {
		    vs = 1.75f + RandomF01() * 1.5f;
			Vector3F base(9.3f * i, 9.3f * j, 0.f * j);
			Vector3F right = base + Vector3F(1.75f, 0.f, 0.f) * vs;
			Vector3F front = base + Vector3F(0.f, 0.f, 1.75f) * vs;
			Vector3F top = base + Vector3F(0.f, 1.75f, 0.f) * vs;
			if((j&1)==0) {
			    right.y = top.y;
			}
			else {
			    base.y -= .5f * vs;
			}
			
			vrx = 0.725f * (RandomF01() - .5f);
			vry = 1.f  * (RandomF01() + 1.f)  * vy;
			vrz = 0.732f * (RandomF01() - .5f);
			vr = 0.13f * RandomF01();
			
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

			unsigned b = (j * GRDW + i) * 4;
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
	
	m_hostPairs->create(GRDW * GRDH / 2 * 8);
	unsigned * pab = (unsigned *)m_hostPairs->data();
	int k = 0;
	for(j=0; j < GRDH; j+=2) {
		for(i=0; i<GRDW; i++) {
		    pab[k*2] = j * GRDW + i;
		    pab[k*2 + 1] = (j+1) * GRDW + i;
		    k++;
		}
	}
	
	Vector3F r0(-1.3f, 1.f, -0.f);
	Vector3F n0(0.f, 1.f, 0.1f);
	n0.normalize();
	Vector3F omega0(0.0f, 0.f, 0.24f);
	
	std::cout<<"\n omega cross r dot n "<<omega0.cross(r0).dot(n0)<<"\n";
	std::cout<<"\n r cross n dot omega "<<r0.cross(n0).dot(omega0)<<"\n";
	
	r0.normalize();
	std::cout<<" omega cross r "<<omega0.cross(r0)<<"\n";
	r0.set(-1.3f, .5f, -0.f);
	r0.normalize();
	std::cout<<" omega cross r "<<omega0.cross(r0)<<"\n";
}

ContactThread::~ContactThread() {}

void ContactThread::initOnDevice()
{
    m_narrowphase->initOnDevice();
	
	m_devicePairs->create(GRDW * GRDH / 2 * 8);
	m_devicePairs->hostToDevice(m_hostPairs->data(), m_hostPairs->bufferSize());
	
	m_narrowphase->computeContacts(m_devicePairs, GRDW * GRDH / 2);
	m_contactSolver->initOnDevice();
}

void ContactThread::stepPhysics(float dt)
{
	m_narrowphase->computeContacts(m_devicePairs, GRDW * GRDH / 2);
	
	m_contactSolver->solveContacts(m_narrowphase->numContacts(),
									m_narrowphase->contactBuffer(),
									m_narrowphase->contactPairsBuffer(),
									m_narrowphase->objectBuffer());
	m_tetra->integrate(0.016667f);
	BaseSolverThread::stepPhysics(dt);
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