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
    connect(m_physicsControl, SIGNAL(youngsModulusChanged(double)), 
            glWidget, SLOT(receiveYoungsModulus(double)));
    
    connect(m_physicsControl, SIGNAL(stiffnessAttenuateEndsChanged(QPointF)), 
            glWidget, SLOT(receiveStiffnessAttenuateEnds(QPointF)));
    
    connect(m_physicsControl, SIGNAL(stiffnessAttenuateLeftChanged(QPointF)), 
            glWidget, SLOT(receiveStiffnessAttenuateLeft(QPointF)));
    
    connect(m_physicsControl, SIGNAL(stiffnessAttenuateRightChanged(QPointF)), 
            glWidget, SLOT(receiveStiffnessAttenuateRight(QPointF)));
    
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
}

void Window::createActions()
{
    showPhysicsControlAct = new QAction(tr("&Physics Control"), this);
	showPhysicsControlAct->setStatusTip(tr("Show physics settings"));
    connect(showPhysicsControlAct, SIGNAL(triggered()), m_physicsControl, SLOT(show()));
}
