#include <QtGui>
#include <QtOpenGL>

#include <math.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include "SceneContainer.h"

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    m_scene = new SceneContainer(getDrawer());
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
    m_scene->renderWorld();
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
			
		}
	}
		
	switch (event->key()) {
		case Qt::Key_A:
		    break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}
//:~
