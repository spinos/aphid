#include <QtGui>

#include <gl_heads.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <CudaBase.h>
#include "DrawNp.h"
#include <CUDABuffer.h>
#include "ContactThread.h"
#include <CudaTetrahedronSystem.h>
#include "SimpleContactSolver.h"

#define SIMULATE_INLINE 1

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

#if SIMULATE_INLINE	
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
#else
	connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
	disconnect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
#endif
}

void GLWidget::clientDraw()
{ 
#if SIMULATE_INLINE
    if(internalTimer()->isActive()) m_solver->stepPhysics(1.f / 60.f);
#endif
    // m_dbgDraw->printTOI(m_solver->narrowphase(), m_solver->hostPairs());
	// m_dbgDraw->printContactPairHash(m_contactSolver, m_narrowphase->numContacts());
	m_solver->tetra()->sendXToHost();
	m_solver->tetra()->sendVToHost();
	m_dbgDraw->drawTetra((TetrahedronSystem *)m_solver->tetra());
	// m_dbgDraw->drawTetraAtFrameEnd((TetrahedronSystem *)m_solver->tetra());
	m_dbgDraw->drawSeparateAxis(m_solver->narrowphase(), m_solver->hostPairs(), (TetrahedronSystem *)m_solver->tetra());
	bool chk = m_dbgDraw->checkConstraint(m_solver->contactSolver(), m_solver->narrowphase(), (TetrahedronSystem *)m_solver->tetra());
	
	if(!chk) internalTimer()->stop();
    //else internalTimer()->start(100);
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
#if SIMULATE_INLINE
            internalTimer()->stop();
#else
			disconnect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
			connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
#endif
			break;
		case Qt::Key_S:
#if SIMULATE_INLINE
            internalTimer()->start();
#else
			disconnect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
			connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
#endif
			break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}
