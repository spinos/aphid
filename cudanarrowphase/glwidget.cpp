#include <QtGui>

#include <gl_heads.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <CudaBase.h>
#include <CudaTetrahedronSystem.h>
#include "DrawNp.h"
#include <CUDABuffer.h>
#include <CudaNarrowphase.h>
#include "SimpleContactSolver.h"

#define GRDX 1226
#define NTET 2500

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_tetra = new CudaTetrahedronSystem;
	m_tetra->create(NTET, 1.f, 1.f);
	float * hv = &m_tetra->hostV()[0];
	
	unsigned i, j;
	float vy = 1.f;
	float vrx, vry, vrz, vr;
	for(j=0; j < 2; j++) {
		for(i=0; i<GRDX; i++) {
			Vector3F base(2.3f * i, 5.f * j + 1.f, 0.5f * i);
			Vector3F right = base + Vector3F(1.75f, 0.f, 0.7f);
			Vector3F front = base + Vector3F(0.f, 0.f, 1.75f);
			Vector3F top = base + Vector3F(0.f, 1.75f, 0.7f);
			if(j%2==0) top.x += 1.75f;
			
			vrx = .5f * (((float)(rand() % 199))/199.f - .5f);
			vry = 1.5f * (((float)(rand() % 199))/199.f + .5f) * vy;
			vrz = .5f * (((float)(rand() % 199))/199.f - .5f);
			vr = .5f * (((float)(rand() % 199))/199.f);
			
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
	
	m_dbgDraw = new DrawNp;
    m_dbgDraw->setDrawer(getDrawer());
	
	m_hostPairs = new BaseBuffer;
	m_devicePairs = new CUDABuffer;
	
	m_hostPairs->create(GRDX * 8);
	unsigned * pab = (unsigned *)m_hostPairs->data();
	for(i=0; i<GRDX; i++) {
		pab[i*2] = i;
		pab[i*2 + 1] = i + GRDX;
	}
	
	m_narrowphase = new CudaNarrowphase;
	m_narrowphase->addTetrahedronSystem(m_tetra);
	
	m_contactSolver = new SimpleContactSolver;
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
	CudaBase::SetDevice();
	
	m_narrowphase->initOnDevice();
	
	m_devicePairs->create(GRDX * 8);
	m_devicePairs->hostToDevice(m_hostPairs->data(), GRDX *8);
	
	m_narrowphase->computeContacts(m_devicePairs, GRDX);
	// connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	// connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
    m_narrowphase->computeContacts(m_devicePairs, GRDX);
	// m_dbgDraw->printTOI(m_narrowphase, m_hostPairs);
	
	m_contactSolver->solveContacts(m_narrowphase->numContacts(),
									m_narrowphase->contacts(),
									m_narrowphase->contactPairsBuffer(),
									m_narrowphase->objectBuffer());
									
	m_dbgDraw->printContactPairHash(m_contactSolver, m_narrowphase->numContacts());
	
	m_tetra->integrate(0.016667f);
	m_tetra->sendXToHost();
	m_dbgDraw->drawTetra(m_tetra);
	//m_dbgDraw->drawTetraAtFrameEnd(m_tetra);
	m_dbgDraw->drawSeparateAxis(m_narrowphase, m_hostPairs, m_tetra);
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
			break;
		case Qt::Key_D:
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
