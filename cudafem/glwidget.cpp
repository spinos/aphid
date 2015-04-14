#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include "CudaDynamicWorld.h"
#include "FEMWorldInterface.h"
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	
	m_world = new CudaDynamicWorld;
	m_interface = new FEMWorldInterface;
	m_interface->create(m_world);
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
    delete m_interface;
    delete m_world;
}

void GLWidget::clientInit()
{	
    m_world->initOnDevice();
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
    if(internalTimer()->isActive())
        m_world->stepPhysics(1.f / 60.f);
    m_interface->draw(m_world, getDrawer());
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

