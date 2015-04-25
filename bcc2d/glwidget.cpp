#include <QtGui>
#include "BccGlobal.h"
#include <gl_heads.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include "DrawNp.h"
#include "BccWorld.h"
#define SIMULATE_INLINE 1

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_world = new BccWorld(getDrawer());
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
    m_world->draw();
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
			m_world->save();
		}
	}
		
	switch (event->key()) {
		case Qt::Key_A:
		    m_world->moveTestP(-.1f, 0.f, 0.f);
			break;
		case Qt::Key_D:
		    m_world->moveTestP(.1f, 0.f, 0.f);
			break;
		case Qt::Key_W:
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
			break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}
