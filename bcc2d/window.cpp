#include <QtGui>
#include "glwidget.h"
#include "window.h"
#include "BccInterface.h"
#include <HesperisFile.h>
#include "GenTetControl.h"

Window::Window()
{
	glWidget = new GLWidget(this);
    m_buildControl = new GenTetControl(this);
    
	setCentralWidget(glWidget);
	
    setWindowTitle(QString("BCC Tetrahedron Mesh - %1").arg(BccInterface::FileName.c_str()));
	
	createActions();
    createMenus();
    
    connect(m_buildControl, SIGNAL(rebuildTet(double)), 
            glWidget, SLOT(receiveRebuildTet(double)));
    
    connect(m_buildControl, SIGNAL(patchMethodChanged(int)), 
            glWidget, SLOT(receivePatchMethod(int)));
    
    connect(glWidget, SIGNAL(estimatedNChanged(unsigned)), 
            m_buildControl, SLOT(receiveEstimatedN(unsigned)));
    
	statusBar()->showMessage(tr("Ready"));
}

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

void Window::createMenus()
{
    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(importTriangleAct);
	fileMenu->addAction(importCurveAct);
	fileMenu->addAction(importPatchAct);
    fileMenu = menuBar()->addMenu(tr("&Window"));
    fileMenu->addAction(buildControlAct);
}

void Window::createActions()
{
    importTriangleAct = new QAction(tr("&Import grow mesh"), this);
	importTriangleAct->setStatusTip(tr("Import a triangle mesh to grow, from a .hes file"));
    connect(importTriangleAct, SIGNAL(triggered()), glWidget, SLOT(importGrowMesh()));
	
	importCurveAct = new QAction(tr("&Import curves"), this);
	importCurveAct->setStatusTip(tr("Import curves from a .hes file"));
    connect(importCurveAct, SIGNAL(triggered()), glWidget, SLOT(importCurve()));
	
	importPatchAct = new QAction(tr("&Import patches"), this);
	importPatchAct->setStatusTip(tr("Import triangle patches to extract curve, from a .hes file"));
    connect(importPatchAct, SIGNAL(triggered()), glWidget, SLOT(importPatch()));
    
    buildControlAct = new QAction(tr("&Building controls"), this);
	buildControlAct->setStatusTip(tr("Control parameters for building tetrahedrons"));
    connect(buildControlAct, SIGNAL(triggered()), m_buildControl, SLOT(show()));
}
//:~