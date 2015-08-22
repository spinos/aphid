#include <QtGui>
#include "glwidget.h"
#include "window.h"
#include "BccInterface.h"
#include <HesperisFile.h>
Window::Window()
{
	glWidget = new GLWidget;

	setCentralWidget(glWidget);
	
    setWindowTitle(QString("BCC Tetrahedron Mesh - %1").arg(BccInterface::FileName.c_str()));
	
	createActions();
    createMenus();
	
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
}

void Window::createActions()
{
    importTriangleAct = new QAction(tr("&Import Grow Mesh"), this);
	importTriangleAct->setStatusTip(tr("Import a triangle mesh to grow, from a .hes file"));
    connect(importTriangleAct, SIGNAL(triggered()), glWidget, SLOT(importGrowMesh()));
	
	importCurveAct = new QAction(tr("&Import Curves"), this);
	importCurveAct->setStatusTip(tr("Import curves from a .hes file"));
    connect(importCurveAct, SIGNAL(triggered()), glWidget, SLOT(importCurve()));
	
	importPatchAct = new QAction(tr("&Import Patches"), this);
	importPatchAct->setStatusTip(tr("Import triangle patches to extract curve, from a .hes file"));
    connect(importPatchAct, SIGNAL(triggered()), glWidget, SLOT(importPatch()));
}
//:~