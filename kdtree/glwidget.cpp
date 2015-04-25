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
		case Qt::Key_I:
			qDebug()<<"up level";
			m_scene->upLevel();
		    break;
		case Qt::Key_M:
			qDebug()<<"down level";
			m_scene->downLevel();
		    break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}
//:~
