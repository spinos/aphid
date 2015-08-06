#include <QtGui>

#include "glwidget.h"
#include "window.h"
#include "PhysicsControl.h"

Window::Window()
{
    glWidget = new GLWidget(this);
	m_physicsControl = new PhysicsControl(this);
	setCentralWidget(glWidget);
    setWindowTitle(tr("Cuda FEM"));
    createActions();
    createMenus();
    connect(m_physicsControl, SIGNAL(densityChanged(double)), 
            glWidget, SLOT(receiveDensity(double)));
	
	connect(m_physicsControl, SIGNAL(youngsModulusChanged(double)), 
            glWidget, SLOT(receiveYoungsModulus(double)));
    
    connect(m_physicsControl, SIGNAL(stiffnessAttenuateEndsChanged(QPointF)), 
            glWidget, SLOT(receiveStiffnessAttenuateEnds(QPointF)));
    
    connect(m_physicsControl, SIGNAL(stiffnessAttenuateLeftChanged(QPointF)), 
            glWidget, SLOT(receiveStiffnessAttenuateLeft(QPointF)));
    
    connect(m_physicsControl, SIGNAL(stiffnessAttenuateRightChanged(QPointF)), 
            glWidget, SLOT(receiveStiffnessAttenuateRight(QPointF)));
    
    connect(m_physicsControl, SIGNAL(windSpeedChanged(double)), 
            glWidget, SLOT(receiveWindSpeed(double)));
    
    connect(m_physicsControl, SIGNAL(windVecChanged(QPointF)), 
            glWidget, SLOT(receiveWindVec(QPointF)));
    
    connect(glWidget, SIGNAL(turnOffCaching()), 
            this, SLOT(receiveTurnOffCaching()));
    
    statusBar()->showMessage(tr("Ready"));
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();

	QWidget::keyPressEvent(e);
}

void Window::createMenus()
{
    windowMenu = menuBar()->addMenu(tr("&Window"));
    windowMenu->addAction(showPhysicsControlAct);
    
    cachingMenu = menuBar()->addMenu(tr("&Caching"));
    cachingMenu->addAction(enableCachingAct);
}

void Window::createActions()
{
    showPhysicsControlAct = new QAction(tr("&Physics Control"), this);
	showPhysicsControlAct->setStatusTip(tr("Show physics settings"));
    connect(showPhysicsControlAct, SIGNAL(triggered()), m_physicsControl, SLOT(show()));
    
    enableCachingAct = new QAction(tr("&Position"), this);
    enableCachingAct->setCheckable(true);
	enableCachingAct->setStatusTip(tr("Enable/Disable writing position cache"));
    connect(enableCachingAct, SIGNAL(triggered()), glWidget, SLOT(togglePositionOut()));
}

void Window::receiveTurnOffCaching()
{ enableCachingAct->setChecked(false); }
//:~
