#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include "LarixWorld.h"
#include "LarixInterface.h"

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_world = new LarixWorld;
	LarixInterface::CreateWorld(m_world);
}

GLWidget::~GLWidget()
{
	delete m_world;
}

void GLWidget::clientInit()
{
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
}

void GLWidget::clientSelect(QMouseEvent * event)
{
	setUpdatesEnabled(false);

	setUpdatesEnabled(true);
}

void GLWidget::clientDeselect(QMouseEvent * event) 
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::clientMouseInput(QMouseEvent * event)
{
	setUpdatesEnabled(false);

	setUpdatesEnabled(true);
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	Base3DView::keyPressEvent(e);
}
