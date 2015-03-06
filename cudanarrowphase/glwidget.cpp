#include <QtGui>

#include <gl_heads.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <CudaBase.h>
#include <CudaTetrahedronSystem.h>
#include "DrawNp.h"
#include <CUDABuffer.h>
#include <CudaNarrowphase.h>

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_tetra = new CudaTetrahedronSystem;
	m_tetra->create(2200, 1.f, 1.f);
	float * hv = &m_tetra->hostV()[0];
	
	unsigned i, j;
	const unsigned grdx = 99;
	float vy = 1.f;
	for(j=0; j < 2; j++) {
		for(i=0; i<grdx; i++) {
			Vector3F base(2.3f * i - 60.f, 3.f * j + 1.f, 0.5f * i);
			Vector3F right = base + Vector3F(1.75f, 0.f, 0.f);
			Vector3F front = base + Vector3F(0.f, 0.f, 1.75f);
			Vector3F top = base + Vector3F(0.f, 1.75f, 0.f);
			if(j%2==0) top.x += 1.75f;
			
			m_tetra->addPoint(&base.x);
			hv[0] = 30.f * (((float)(rand() % 199))/199.f - .5f);
			hv[1] = 30.f * (((float)(rand() % 199))/199.f) * vy;
			hv[2] = 33.f * (((float)(rand() % 199))/199.f - .5f);
			hv+=3;
			m_tetra->addPoint(&right.x);
			hv[0] = 30.f * (((float)(rand() % 199))/199.f - .5f);
			hv[1] = 38.f * (((float)(rand() % 199))/199.f) * vy;
			hv[2] = 35.f * (((float)(rand() % 199))/199.f - .5f);
			hv+=3;
			m_tetra->addPoint(&top.x);
			hv[0] = 30.f * (((float)(rand() % 199))/199.f - .5f);
			hv[1] = 38.f * (((float)(rand() % 199))/199.f) * vy;
			hv[2] = 35.f * (((float)(rand() % 199))/199.f - .5f);
			hv+=3;
			m_tetra->addPoint(&front.x);
			hv[0] = 30.f * (((float)(rand() % 199))/199.f - .5f);
			hv[1] = 38.f * (((float)(rand() % 199))/199.f) * vy;
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
		vy = -vy;
	}
	
	qDebug()<<"tetra n tet "<<m_tetra->numTetradedrons();
	qDebug()<<"tetra n p "<<m_tetra->numPoints();
	qDebug()<<"tetra n tri "<<m_tetra->numTriangles();
	
	m_dbgDraw = new DrawNp;
    m_dbgDraw->setDrawer(getDrawer());
	
	m_hostPairs = new BaseBuffer;
	m_devicePairs = new CUDABuffer;
	
	m_hostPairs->create(99 * 8);
	unsigned * pab = (unsigned *)m_hostPairs->data();
	for(i=0; i<grdx; i++) {
		pab[i*2] = i;
		pab[i*2 + 1] = i + 99;
	}
	
	m_narrowphase = new CudaNarrowphase;
	m_narrowphase->addTetrahedronSystem(m_tetra);
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
	CudaBase::SetDevice();
	
	m_tetra->initOnDevice();
	m_narrowphase->initOnDevice();
	
	m_devicePairs->create(99 * 8);
	m_devicePairs->hostToDevice(m_hostPairs->data(), 99 *8);
	
	m_narrowphase->computeContacts(m_devicePairs, 99);
	m_dbgDraw->printCoord(m_narrowphase, m_hostPairs);
	// connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	// connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	m_dbgDraw->drawTetra(m_tetra);
	m_dbgDraw->drawTetraAtFrameEnd(m_tetra);
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
