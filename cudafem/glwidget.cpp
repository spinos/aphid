#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include "CudaDynamicWorld.h"
#include "FEMWorldInterface.h"
#include <WorldThread.h>
#include <WorldDbgDraw.h>

#define DRGDRW 0
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    std::cout<<"\n press R to reset"
    <<"\n press Space to toggle physics"
    <<"\n press L to enable bake file"
    <<"\n";
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	
	m_world = new CudaDynamicWorld;
	m_interface = new FEMWorldInterface;
	m_interface->create(m_world);
	
	m_thread = new WorldThread(m_world, this);
	m_isPhysicsRunning = 0;
}

GLWidget::~GLWidget()
{
    delete m_thread;
    delete m_interface;
    delete m_world;
}

void GLWidget::clientInit()
{	
    CudaDynamicWorld::DbgDrawer = new WorldDbgDraw(getDrawer());
    m_world->initOnDevice();
    startPhysics();
}

void GLWidget::clientDraw()
{
    if(m_thread->numLoops() < WorldThread::NumSubsteps) {
        emit updatePhysics();
        return;
    }
	else if(m_isPhysicsRunning)
        emit updatePhysics();
    
    std::stringstream sst;
	sst.str("");
	sst<<"fps: "<<frameRate();
    hudText(sst.str(), 1);
    sst.str("");
	sst<<"n contacts: "<<m_world->numContacts();
    hudText(sst.str(), 2);
    
    if(m_interface->HasVelocityFile()) {
        sst.str("");
        sst<<"cache range: ("<<m_interface->VelocityFileBegin()
        <<","<<m_interface->VelocityFileEnd()<<")";
        hudText(sst.str(), 3);
        sst.str("");
        sst<<"current frame: "<<m_interface->CurrentFrame();
        hudText(sst.str(), 4);
        if(m_world->isToSaveCache()) {
            if(m_world->allFramesCached())
                hudText("all frames cached", 5);
            else  
                hudText("cacheing frames", 5);
        }
    }
    
#if DRGDRW
    if(m_isPhysicsRunning) m_interface->draw(m_world, getDrawer());
#else
    m_interface->draw(m_world, getDrawer());
    // m_world->dbgDraw();

#endif
    //else m_interface->drawFaulty(m_world, getDrawer());
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
#if DRGDRW
	for(int i=0; i < 2; i++) {
	    m_world->collide();
	    if(!m_interface->verifyData(m_world)) 
	        stopPhysics();
	    else 
	        m_world->integrate(1.f / 60.f);
    }
    m_world->sendXToHost();
#endif
	update();
}

void GLWidget::stopPhysics()
{
#if DRGDRW
    disconnect(internalTimer(), SIGNAL(timeout()), this, SLOT(simulate()));
#else
    disconnect(internalTimer(), SIGNAL(timeout()), m_thread, SLOT(simulate()));
    disconnect(m_thread, SIGNAL(doneStep()), this, SLOT(update()));
#endif
    connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
    m_isPhysicsRunning = false;
}

void GLWidget::startPhysics()
{
#if DRGDRW
    connect(internalTimer(), SIGNAL(timeout()), this, SLOT(simulate()));
#else
    connect(this, SIGNAL(updatePhysics()), m_thread, SLOT(simulate()));
    connect(m_thread, SIGNAL(doneStep()), this, SLOT(update()));
#endif
    m_isPhysicsRunning = true;
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
        case Qt::Key_R:
            m_world->reset();
            emit turnOffCaching();
            break;
        case Qt::Key_Space:
            togglePhysics();
            break;
        case Qt::Key_L:
			m_interface->useVelocityFile(m_world);
            break;
        case Qt::Key_A:
			m_interface->changeMaxDisplayLevel(1);
			break;
		case Qt::Key_D:
			m_interface->changeMaxDisplayLevel(-1);
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

void GLWidget::togglePhysics()
{
    if(m_isPhysicsRunning) stopPhysics();
    else startPhysics();   
}

void GLWidget::receiveDensity(double x)
{ m_interface->updateDensity((float)x); }

void GLWidget::receiveYoungsModulus(double x)
{ m_interface->updateYoungsModulus((float)x); }

void GLWidget::receiveStiffnessAttenuateEnds(QPointF v)
{ m_interface->updateStiffnessMapEnds(v.x(), v.y()); }

void GLWidget::receiveStiffnessAttenuateLeft(QPointF v)
{ m_interface->updateStiffnessMapLeft(v.x(), v.y()); }

void GLWidget::receiveStiffnessAttenuateRight(QPointF v)
{ m_interface->updateStiffnessMapRight(v.x(), v.y()); }

void GLWidget::receiveWindSpeed(double x)
{ m_interface->updateWindSpeed(x); }

void GLWidget::receiveWindVec(QPointF v)
{ 
    Vector3F cat = Vector3F::FromPolarXYZ(v.x(), v.y());
    m_interface->updateWindVec(cat); 
}

void GLWidget::togglePositionOut()
{
    if(m_world->isToSaveCache()) m_world->setToSaveCache(false);
    else m_world->setToSaveCache(true);
}
//:~
