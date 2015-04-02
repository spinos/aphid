#include <QtGui>

#include <gl_heads.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <CudaBase.h>
#include "DrawNp.h"
#include <CUDABuffer.h>
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
	switch (event->key()) {
		case Qt::Key_A:
			break;
		case Qt::Key_D:
			break;
		case Qt::Key_W:
#if SIMULATE_INLINE
            internalTimer()->stop();
#else
#endif
			break;
		case Qt::Key_S:
#if SIMULATE_INLINE
            internalTimer()->start();
#else
#endif
			break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}
