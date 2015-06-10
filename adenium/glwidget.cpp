#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include "AdeniumInterface.h"
#include "AdeniumWorld.h"
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
    connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
    m_world = new AdeniumWorld;
    AdeniumInterface adei;
    adei.create(m_world);
}

GLWidget::~GLWidget()
{
    delete m_world;
}

void GLWidget::clientInit()
{	
    m_world->initOnDevice();
}

void GLWidget::clientDraw()
{
    m_world->draw();
}
//! [7]

//! [9]
void GLWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}
//! [9]

void GLWidget::clientDeselect()
{
}

//! [10]
void GLWidget::clientMouseInput(Vector3F & stir)
{
}
//! [10]

void GLWidget::simulate()
{
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}

void GLWidget::resizeEvent(QResizeEvent * event)
{
	QSize renderAreaSize = size();
    //qDebug()<<"render size "<<renderAreaSize.width()<<" "<<renderAreaSize.height();
    m_world->resizeRenderArea(renderAreaSize.width(), renderAreaSize.height());
    Base3DView::resizeEvent(event);
}
