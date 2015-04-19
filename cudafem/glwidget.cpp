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
    startPhysics();
}

void GLWidget::clientDraw()
{
    if(m_isPhysicsRunning) m_interface->draw(m_world, getDrawer());
    else m_interface->drawFaulty(m_world, getDrawer());
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
	for(int i=0; i < 2; i++) {
    m_world->collide();
    if(!m_interface->verifyData(m_world)) 
        stopPhysics();
    else
        m_world->integrate(1.f / 60.f);
	}
	update();
}

void GLWidget::stopPhysics()
{
    disconnect(internalTimer(), SIGNAL(timeout()), this, SLOT(simulate()));
    connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
    m_isPhysicsRunning = false;
}

void GLWidget::startPhysics()
{
    disconnect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
    connect(internalTimer(), SIGNAL(timeout()), this, SLOT(simulate()));
    m_isPhysicsRunning = true;
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
        case Qt::Key_R:
            m_world->reset();
            break;
        case Qt::Key_S:
            if(m_isPhysicsRunning) 
                stopPhysics();
            else
                startPhysics();
            break;
        default:
			break;
    }
	Base3DView::keyPressEvent(event);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}

