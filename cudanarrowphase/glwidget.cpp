#include <QtGui>

#include <gl_heads.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <CudaBase.h>
#include "DrawNp.h"
#include <CUDABuffer.h>
#include "ContactThread.h"
#include <CudaTetrahedronSystem.h>
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    m_dbgDraw = new DrawNp;
    m_dbgDraw->setDrawer(getDrawer());
	m_solver = new ContactThread;
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
	CudaBase::SetDevice();
	
	m_solver->initOnDevice();
	// connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
	
	connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{ 								
	// m_dbgDraw->printContactPairHash(m_contactSolver, m_narrowphase->numContacts());
	m_solver->tetra()->sendXToHost();
	m_dbgDraw->drawTetra((TetrahedronSystem *)m_solver->tetra());
	//m_dbgDraw->drawTetraAtFrameEnd(m_tetra);
	m_dbgDraw->drawSeparateAxis(m_solver->narrowphase(), m_solver->hostPairs(), (TetrahedronSystem *)m_solver->tetra());
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
