#include <QtGui>
#include "BccGlobal.h"
#include <gl_heads.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include "DrawNp.h"
#include "BccWorld.h"
#include "FitTest.h"

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
#if TEST_FIT
    FitBccMeshBuilder::EstimatedGroupSize = 2.1f;
	m_fit = new FitTest(getDrawer());
#else
	m_world = new BccWorld(getDrawer());
#endif
}

GLWidget::~GLWidget()
{ 
}

void GLWidget::clientInit()
{
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
#if TEST_FIT
	m_fit->draw();
#else
	m_world->draw();
#endif
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
	if(event->modifiers() == Qt::ControlModifier | Qt::MetaModifier) {
		if(event->key() == Qt::Key_S) {
#if TEST_FIT

#else
			m_world->save();
#endif
		}
	}
	
	switch (event->key()) {
		case Qt::Key_K:
		    m_world->rebuildTetrahedronsMesh(-1.f);
			break;
		case Qt::Key_L:
		    m_world->rebuildTetrahedronsMesh(1.f);
			break;
		/*case Qt::Key_W:
		    m_world->moveTestP(0.f, .1f, 0.f);
			break;
		case Qt::Key_S:
		    m_world->moveTestP(0.f, -.1f, 0.f);
			break;
		case Qt::Key_F:
		    m_world->moveTestP(0.f, 0.f, .1f);
			break;
		case Qt::Key_B:
		    m_world->moveTestP(0.f, 0.f, -.1f);
			break;*/
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}
